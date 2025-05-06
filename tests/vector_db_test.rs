use anyhow::Result;
use mcp_chatbot::vector_db::{VectorDBClient, VectorDBConfig};
use qdrant_client::qdrant::{value, Value};
use serde_json::json;
use std::process::Command;
use std::time::Duration;
use tokio;
use tokio::time::sleep;

const QDRANT_DOCKER_IMAGE: &str = "qdrant/qdrant:latest";
const QDRANT_CONTAINER_NAME: &str = "qdrant_test";
const QDRANT_STARTUP_WAIT: u64 = 5;

// 将 serde_json::Value 转换为 qdrant_client::qdrant::Value
fn convert_to_qdrant_value(json_value: serde_json::Value) -> Value {
    match json_value {
        serde_json::Value::String(s) => Value {
            kind: Some(value::Kind::StringValue(s)),
        },
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value {
                    kind: Some(value::Kind::IntegerValue(i)),
                }
            } else if let Some(f) = n.as_f64() {
                Value {
                    kind: Some(value::Kind::DoubleValue(f)),
                }
            } else {
                Value {
                    kind: Some(value::Kind::StringValue(n.to_string())),
                }
            }
        }
        serde_json::Value::Bool(b) => Value {
            kind: Some(value::Kind::BoolValue(b)),
        },
        serde_json::Value::Null => Value {
            kind: Some(value::Kind::NullValue(())),
        },
        serde_json::Value::Array(arr) => Value {
            kind: Some(value::Kind::ListValue(qdrant_client::qdrant::ListValue {
                values: arr.into_iter().map(convert_to_qdrant_value).collect(),
            })),
        },
        serde_json::Value::Object(obj) => Value {
            kind: Some(value::Kind::StructValue(qdrant_client::qdrant::Struct {
                fields: obj
                    .into_iter()
                    .map(|(k, v)| (k, convert_to_qdrant_value(v)))
                    .collect(),
            })),
        },
    }
}

async fn setup_qdrant() -> Result<()> {
    // 停止并删除已存在的容器
    let _ = Command::new("docker")
        .args(["stop", QDRANT_CONTAINER_NAME])
        .output();
    let _ = Command::new("docker")
        .args(["rm", QDRANT_CONTAINER_NAME])
        .output();

    // 启动新的 Qdrant 容器
    let output = Command::new("docker")
        .args([
            "run",
            "-d",
            "--name",
            QDRANT_CONTAINER_NAME,
            "-p",
            "6333:6333",
            "-p",
            "6334:6334",
            QDRANT_DOCKER_IMAGE,
        ])
        .output()?;

    if !output.status.success() {
        let error = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!(
            "Failed to start Qdrant container: {}",
            error
        ));
    }

    // 等待 Qdrant 服务启动
    sleep(Duration::from_secs(QDRANT_STARTUP_WAIT)).await;

    // 验证容器是否正在运行
    let status = Command::new("docker")
        .args(["ps", "-q", "-f", &format!("name={}", QDRANT_CONTAINER_NAME)])
        .output()?;

    if status.stdout.is_empty() {
        return Err(anyhow::anyhow!("Qdrant container failed to start"));
    }

    Ok(())
}

async fn teardown_qdrant() -> Result<()> {
    let stop_output = Command::new("docker")
        .args(["stop", QDRANT_CONTAINER_NAME])
        .output()?;

    if !stop_output.status.success() {
        let error = String::from_utf8_lossy(&stop_output.stderr);
        return Err(anyhow::anyhow!(
            "Failed to stop Qdrant container: {}",
            error
        ));
    }

    let rm_output = Command::new("docker")
        .args(["rm", QDRANT_CONTAINER_NAME])
        .output()?;

    if !rm_output.status.success() {
        let error = String::from_utf8_lossy(&rm_output.stderr);
        return Err(anyhow::anyhow!(
            "Failed to remove Qdrant container: {}",
            error
        ));
    }

    Ok(())
}

#[tokio::test]
async fn test_vector_db_basic_operations() -> Result<()> {
    // 设置 Qdrant Docker 容器
    setup_qdrant().await?;

    // 创建测试配置
    let config = VectorDBConfig {
        provider: "qdrant".to_string(),
        host: "localhost".to_string(),
        port: 6333,
        http_port: 6334,
        collection_name: "test_vectors".to_string(),
        vector_size: 4,
        distance: "Cosine".to_string(),
        on_disk_payload: true,
        optimizers_config: mcp_chatbot::vector_db::OptimizersConfig {
            default_segment_number: 2,
            memmap_threshold: 20000,
        },
        wal_config: mcp_chatbot::vector_db::WalConfig {
            wal_capacity_mb: 32,
            wal_segments_capacity_mb: 64,
        },
        performance_config: mcp_chatbot::vector_db::PerformanceConfig {
            max_search_threads: 4,
            max_optimization_threads: 2,
        },
    };

    // 初始化客户端
    let client = VectorDBClient::new(config)?;

    // 初始化集合
    client.init().await?;

    // 测试向量插入
    let test_vectors = vec![
        (
            "test_id_1".to_string(),
            vec![1.0, 2.0, 3.0, 4.0],
            json!({"text": "test vector 1"}),
        ),
        (
            "test_id_2".to_string(),
            vec![2.0, 3.0, 4.0, 5.0],
            json!({"text": "test vector 2"}),
        ),
    ];
    client.upsert_vectors(test_vectors).await?;

    // 测试向量搜索
    let search_vector = vec![1.0, 2.0, 3.0, 4.0];
    let results = client.search_vectors(search_vector, 2).await?;

    // 验证搜索结果
    assert!(!results.is_empty(), "Search results should not be empty");
    assert_eq!(results.len(), 2, "Should return 2 results");

    // 验证第一个结果应该是与搜索向量最相似的
    let (first_id, first_score, _) = &results[0];
    assert_eq!(first_id, "test_id_1", "First result should be test_id_1");
    assert!(
        *first_score > 0.9_f32,
        "Similarity score should be high for exact match"
    );

    // 清理 Qdrant Docker 容器
    teardown_qdrant().await?;

    Ok(())
}

#[tokio::test]
async fn test_vector_db_error_handling() -> Result<()> {
    // 设置 Qdrant Docker 容器
    setup_qdrant().await?;

    // 测试无效配置
    let invalid_config = VectorDBConfig {
        provider: "invalid".to_string(),
        host: "localhost".to_string(),
        port: 6333,
        http_port: 6334,
        collection_name: "test_vectors_error".to_string(),
        vector_size: 4,
        distance: "Invalid".to_string(),
        on_disk_payload: true,
        optimizers_config: mcp_chatbot::vector_db::OptimizersConfig {
            default_segment_number: 2,
            memmap_threshold: 20000,
        },
        wal_config: mcp_chatbot::vector_db::WalConfig {
            wal_capacity_mb: 32,
            wal_segments_capacity_mb: 64,
        },
        performance_config: mcp_chatbot::vector_db::PerformanceConfig {
            max_search_threads: 4,
            max_optimization_threads: 2,
        },
    };

    // 验证无效配置会返回错误
    let result = VectorDBClient::new(invalid_config);
    assert!(result.is_err(), "Invalid config should return error");

    // 清理 Qdrant Docker 容器
    teardown_qdrant().await?;

    Ok(())
}

#[tokio::test]
async fn test_vector_db_empty_search() -> Result<()> {
    // 设置 Qdrant Docker 容器
    setup_qdrant().await?;

    let config = VectorDBConfig {
        provider: "qdrant".to_string(),
        host: "localhost".to_string(),
        port: 6333,
        http_port: 6334,
        collection_name: "test_vectors_empty".to_string(),
        vector_size: 4,
        distance: "Cosine".to_string(),
        on_disk_payload: true,
        optimizers_config: mcp_chatbot::vector_db::OptimizersConfig {
            default_segment_number: 2,
            memmap_threshold: 20000,
        },
        wal_config: mcp_chatbot::vector_db::WalConfig {
            wal_capacity_mb: 32,
            wal_segments_capacity_mb: 64,
        },
        performance_config: mcp_chatbot::vector_db::PerformanceConfig {
            max_search_threads: 4,
            max_optimization_threads: 2,
        },
    };

    let client = VectorDBClient::new(config)?;
    client.init().await?;

    // 测试空集合的搜索
    let search_vector = vec![1.0, 2.0, 3.0, 4.0];
    let results = client.search_vectors(search_vector, 1).await?;
    assert!(
        results.is_empty(),
        "Search results should be empty for new collection"
    );

    // 清理 Qdrant Docker 容器
    teardown_qdrant().await?;

    Ok(())
}
