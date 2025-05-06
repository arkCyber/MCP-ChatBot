use anyhow::Result;
use log::info;
use qdrant_client::{
    qdrant::{
        point_id, value, vectors, vectors_config::Config, CreateCollection, Distance,
        OptimizersConfigDiff, PointId, PointStruct, SearchPoints, SearchPointsBuilder,
        UpsertPointsBuilder, Value, Vector, VectorParams, Vectors, VectorsConfig, WalConfigDiff,
    },
    Qdrant,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct VectorDBConfig {
    pub provider: String,
    pub host: String,
    pub port: u32,
    pub http_port: u32,
    pub collection_name: String,
    pub vector_size: u64,
    pub distance: String,
    pub on_disk_payload: bool,
    pub optimizers_config: OptimizersConfig,
    pub wal_config: WalConfig,
    pub performance_config: PerformanceConfig,
}

#[derive(Debug, Deserialize)]
pub struct OptimizersConfig {
    pub default_segment_number: u32,
    pub memmap_threshold: u32,
}

#[derive(Debug, Deserialize)]
pub struct WalConfig {
    pub wal_capacity_mb: u32,
    pub wal_segments_capacity_mb: u32,
}

#[derive(Debug, Deserialize)]
pub struct PerformanceConfig {
    pub max_search_threads: u32,
    pub max_optimization_threads: u32,
}

pub struct VectorDBClient {
    client: Qdrant,
    config: VectorDBConfig,
}

impl VectorDBClient {
    pub fn new(config: VectorDBConfig) -> Result<Self> {
        let client =
            Qdrant::from_url(&format!("http://{}:{}", config.host, config.http_port)).build()?;

        Ok(Self { client, config })
    }

    pub async fn init(&self) -> Result<()> {
        // 检查集合是否存在
        let collections = self.client.list_collections().await?;
        let collection_exists = collections
            .collections
            .iter()
            .any(|c| c.name == self.config.collection_name);

        if !collection_exists {
            info!("Creating collection: {}", self.config.collection_name);
            self.create_collection().await?;
        } else {
            info!("Collection {} already exists", self.config.collection_name);
        }

        Ok(())
    }

    async fn create_collection(&self) -> Result<()> {
        let distance = match self.config.distance.as_str() {
            "Cosine" => Distance::Cosine,
            "Euclid" => Distance::Euclid,
            "Dot" => Distance::Dot,
            _ => Distance::Cosine,
        };

        let vectors_config = VectorsConfig {
            config: Some(Config::Params(VectorParams {
                size: self.config.vector_size,
                distance: distance.into(),
                ..Default::default()
            })),
        };

        let optimizers_config = OptimizersConfigDiff {
            default_segment_number: Some(
                self.config.optimizers_config.default_segment_number as u64,
            ),
            memmap_threshold: Some(self.config.optimizers_config.memmap_threshold as u64),
            ..Default::default()
        };

        let wal_config = WalConfigDiff {
            wal_capacity_mb: Some(self.config.wal_config.wal_capacity_mb as u64),
            wal_segments_ahead: Some(self.config.wal_config.wal_segments_capacity_mb as u64),
            ..Default::default()
        };

        self.client
            .create_collection(
                qdrant_client::qdrant::CreateCollectionBuilder::new(&self.config.collection_name)
                    .vectors_config(vectors_config)
                    .optimizers_config(optimizers_config)
                    .wal_config(wal_config)
                    .on_disk_payload(self.config.on_disk_payload),
            )
            .await?;

        Ok(())
    }

    pub async fn upsert_vectors(
        &self,
        points: Vec<(String, Vec<f32>, serde_json::Value)>,
    ) -> Result<()> {
        let points: Vec<PointStruct> = points
            .into_iter()
            .map(|(id, vector, payload)| PointStruct {
                id: Some(PointId {
                    point_id_options: Some(point_id::PointIdOptions::Uuid(id)),
                }),
                vectors: Some(Vectors {
                    vectors_options: Some(vectors::VectorsOptions::Vector(Vector {
                        data: vector,
                        indices: None,
                        vector: None,
                        vectors_count: None,
                    })),
                }),
                payload: serde_json::to_value(payload)
                    .unwrap_or_default()
                    .as_object()
                    .map(|m| {
                        m.iter()
                            .map(|(k, v)| {
                                (
                                    k.clone(),
                                    Value {
                                        kind: Some(value::Kind::StringValue(v.to_string())),
                                    },
                                )
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
            })
            .collect();

        self.client
            .upsert_points(
                UpsertPointsBuilder::new(&self.config.collection_name, points)
                    .wait(true)
                    .build(),
            )
            .await?;

        Ok(())
    }

    pub async fn search_vectors(
        &self,
        vector: Vec<f32>,
        limit: u32,
    ) -> Result<Vec<(String, f32, serde_json::Value)>> {
        let search_result = self
            .client
            .search_points(
                SearchPointsBuilder::new(&self.config.collection_name, vector, limit as u64)
                    .with_payload(true),
            )
            .await?;

        let results = search_result
            .result
            .into_iter()
            .map(|scored_point| {
                let id = match scored_point.id.unwrap().point_id_options {
                    Some(point_id::PointIdOptions::Uuid(id)) => id,
                    _ => "".to_string(),
                };

                let score = scored_point.score;
                let payload = scored_point
                    .payload
                    .iter()
                    .map(|(k, v)| {
                        let value = match &v.kind {
                            Some(value::Kind::StringValue(s)) => s.clone(),
                            _ => "".to_string(),
                        };
                        (k.clone(), serde_json::Value::String(value))
                    })
                    .collect::<serde_json::Map<String, serde_json::Value>>();

                (id, score, serde_json::Value::Object(payload))
            })
            .collect();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn create_test_config() -> VectorDBConfig {
        VectorDBConfig {
            provider: "qdrant".to_string(),
            host: "localhost".to_string(),
            port: 6333,
            http_port: 6334,
            collection_name: "test_collection".to_string(),
            vector_size: 4,
            distance: "Cosine".to_string(),
            on_disk_payload: true,
            optimizers_config: OptimizersConfig {
                default_segment_number: 2,
                memmap_threshold: 20000,
            },
            wal_config: WalConfig {
                wal_capacity_mb: 32,
                wal_segments_capacity_mb: 64,
            },
            performance_config: PerformanceConfig {
                max_search_threads: 4,
                max_optimization_threads: 2,
            },
        }
    }

    #[tokio::test]
    async fn test_vector_db_operations() {
        let config = create_test_config();
        let client = VectorDBClient::new(config).unwrap();
        client.init().await.unwrap();

        // Test upsert
        let points = vec![(
            "test_id".to_string(),
            vec![1.0, 2.0, 3.0, 4.0],
            json!({"text": "test"}),
        )];
        client.upsert_vectors(points).await.unwrap();

        // Test search
        let search_vector = vec![1.0, 2.0, 3.0, 4.0];
        let results = client.search_vectors(search_vector, 1).await.unwrap();
        assert!(!results.is_empty());
    }
}
