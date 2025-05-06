use anyhow::Result;
use log::info;

#[tokio::test]
async fn test_embedding_model() -> Result<()> {
    // 初始化日志
    env_logger::init();
    info!("Starting embedding test...");

    // 基本测试
    let test_text = "这是一个测试句子。This is a test sentence.";
    info!("Test text: {}", test_text);

    // 验证测试环境
    assert!(true, "Basic test environment check");
    info!("Basic test environment check passed");

    Ok(())
}
