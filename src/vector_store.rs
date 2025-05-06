use anyhow::{Context, Result};
use qdrant_client::{
    qdrant::{
        points_selector::PointsSelectorOneOf, CreateCollection, DeletePointsBuilder, Distance,
        GetPointsBuilder, PointStruct, PointsIdsList, PointsSelector, SearchPoints,
        SearchPointsBuilder, UpsertPointsBuilder, Value, VectorParams, Vectors, VectorsConfig,
        WithPayloadSelector, WithVectorsSelector, WriteOrdering,
    },
    Qdrant,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, thiserror::Error)]
pub enum VectorStoreError {
    #[error("Collection error: {0}")]
    CollectionError(String),
    #[error("Storage error: {0}")]
    StorageError(String),
    #[error("Search error: {0}")]
    SearchError(String),
    #[error("Invalid dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: u64, actual: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub role: String,
    pub content: String,
    pub timestamp: i64,
    pub metadata: HashMap<String, String>,
}

pub struct VectorStore {
    client: Arc<Qdrant>,
    collection_name: String,
    dimension: u64,
}

impl VectorStore {
    pub async fn new(url: &str, collection_name: &str, dimension: u64) -> Result<Self> {
        let client = Arc::new(Qdrant::from_url(url).build()?);

        // Check if collection exists, create if not
        let collections = client.list_collections().await?;
        if !collections
            .collections
            .iter()
            .any(|c| c.name == collection_name)
        {
            client
                .create_collection(
                    qdrant_client::qdrant::CreateCollectionBuilder::new(collection_name)
                        .vectors_config(VectorsConfig {
                            config: Some(qdrant_client::qdrant::vectors_config::Config::Params(
                                VectorParams {
                                    size: dimension as u64,
                                    distance: Distance::Cosine.into(),
                                    ..Default::default()
                                },
                            )),
                        }),
                )
                .await
                .context("Failed to create collection")?;
        }

        Ok(Self {
            client,
            collection_name: collection_name.to_string(),
            dimension,
        })
    }

    pub async fn store_message(&self, message: Message, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() as u64 != self.dimension {
            return Err(anyhow::anyhow!(VectorStoreError::InvalidDimension {
                expected: self.dimension,
                actual: embedding.len() as u64,
            }));
        }

        let mut payload = message.metadata.clone();
        payload.insert("role".to_string(), message.role.clone());
        payload.insert("content".to_string(), message.content.clone());
        payload.insert("timestamp".to_string(), message.timestamp.to_string());

        let point = PointStruct {
            id: Some(message.id.clone().into()),
            vectors: Some(Vectors::from(embedding)),
            payload: payload
                .into_iter()
                .map(|(k, v)| (k, Value::from(v)))
                .collect(),
        };

        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection_name, vec![point]).wait(true))
            .await
            .context("Failed to store message")?;

        Ok(())
    }

    pub async fn store_messages_batch(&self, messages: Vec<(Message, Vec<f32>)>) -> Result<()> {
        let points: Vec<PointStruct> = messages
            .into_iter()
            .map(|(message, embedding)| {
                let mut payload = message.metadata.clone();
                payload.insert("role".to_string(), message.role.clone());
                payload.insert("content".to_string(), message.content.clone());
                payload.insert("timestamp".to_string(), message.timestamp.to_string());

                PointStruct {
                    id: Some(message.id.clone().into()),
                    vectors: Some(Vectors::from(embedding)),
                    payload: payload
                        .into_iter()
                        .map(|(k, v)| (k, Value::from(v)))
                        .collect(),
                }
            })
            .collect();

        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection_name, points).wait(true))
            .await
            .context("Failed to store messages batch")?;

        Ok(())
    }

    pub async fn search_similar(
        &self,
        embedding: Vec<f32>,
        limit: u64,
    ) -> Result<Vec<(String, f32, HashMap<String, String>)>> {
        if embedding.len() as u64 != self.dimension {
            return Err(anyhow::anyhow!(VectorStoreError::InvalidDimension {
                expected: self.dimension,
                actual: embedding.len() as u64,
            }));
        }

        let search_result = self
            .client
            .search_points(
                SearchPointsBuilder::new(&self.collection_name, embedding, limit)
                    .with_payload(true),
            )
            .await
            .context("Failed to search points")?;

        let results = search_result
            .result
            .into_iter()
            .map(|point| {
                let id = point.id.map(|id| format!("{:?}", id)).unwrap_or_default();
                let score = point.score;
                let payload = point
                    .payload
                    .into_iter()
                    .map(|(k, v)| (k, v.to_string()))
                    .collect();
                (id, score, payload)
            })
            .collect();

        Ok(results)
    }

    pub async fn search_similar_batch(
        &self,
        embeddings: Vec<Vec<f32>>,
        limit: u64,
    ) -> Result<Vec<Vec<(String, f32, HashMap<String, String>)>>> {
        // Validate dimensions
        for embedding in &embeddings {
            if embedding.len() as u64 != self.dimension {
                return Err(anyhow::anyhow!(VectorStoreError::InvalidDimension {
                    expected: self.dimension,
                    actual: embedding.len() as u64,
                }));
            }
        }

        let mut results = Vec::with_capacity(embeddings.len());
        for embedding in embeddings {
            let search_result = self
                .client
                .search_points(
                    SearchPointsBuilder::new(&self.collection_name, embedding, limit)
                        .with_payload(true),
                )
                .await
                .context("Failed to search points")?;

            let batch_results = search_result
                .result
                .into_iter()
                .map(|point| {
                    let id = point.id.map(|id| format!("{:?}", id)).unwrap_or_default();
                    let score = point.score;
                    let payload = point
                        .payload
                        .into_iter()
                        .map(|(k, v)| (k, v.to_string()))
                        .collect();
                    (id, score, payload)
                })
                .collect();

            results.push(batch_results);
        }

        Ok(results)
    }

    pub async fn delete_message(&self, message_id: &str) -> Result<()> {
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection_name)
                    .points(PointsIdsList {
                        ids: vec![message_id.into()],
                    })
                    .wait(true),
            )
            .await
            .context("Failed to delete message")?;

        Ok(())
    }

    pub async fn delete_messages_batch(&self, message_ids: Vec<String>) -> Result<()> {
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection_name)
                    .points(PointsIdsList {
                        ids: message_ids.into_iter().map(|id| id.into()).collect(),
                    })
                    .wait(true),
            )
            .await
            .context("Failed to delete messages batch")?;

        Ok(())
    }

    pub async fn get_message(&self, message_id: &str) -> Result<Option<Message>> {
        let result = self
            .client
            .get_points(
                GetPointsBuilder::new(&self.collection_name, vec![message_id.into()])
                    .with_payload(true),
            )
            .await
            .context("Failed to get message")?;

        if result.result.is_empty() {
            return Ok(None);
        }

        let point = &result.result[0];
        let payload = &point.payload;

        let role = payload
            .get("role")
            .map(|v| v.to_string())
            .unwrap_or_default();
        let content = payload
            .get("content")
            .map(|v| v.to_string())
            .unwrap_or_default();
        let timestamp = payload
            .get("timestamp")
            .and_then(|v| v.to_string().parse().ok())
            .unwrap_or_default();

        let mut metadata = HashMap::new();
        for (k, v) in payload {
            if k != "role" && k != "content" && k != "timestamp" {
                metadata.insert(k.clone(), v.to_string());
            }
        }

        Ok(Some(Message {
            id: point
                .id
                .as_ref()
                .map(|id| format!("{:?}", id))
                .unwrap_or_default(),
            role,
            content,
            timestamp,
            metadata,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    async fn create_test_store() -> Result<VectorStore> {
        VectorStore::new("http://localhost:6333", "test_collection", 384).await
    }

    #[tokio::test]
    async fn test_store_and_retrieve_message() -> Result<()> {
        let store = create_test_store().await?;

        let message = Message {
            id: uuid::Uuid::new_v4().to_string(),
            role: "user".to_string(),
            content: "Hello, world!".to_string(),
            timestamp: Utc::now().timestamp(),
            metadata: HashMap::new(),
        };

        let embedding = vec![0.1; 384];
        store
            .store_message(message.clone(), embedding.clone())
            .await?;

        let retrieved = store.get_message(&message.id).await?;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, message.content);

        Ok(())
    }

    #[tokio::test]
    async fn test_search_similar() -> Result<()> {
        let store = create_test_store().await?;

        let message = Message {
            id: uuid::Uuid::new_v4().to_string(),
            role: "user".to_string(),
            content: "Hello, world!".to_string(),
            timestamp: Utc::now().timestamp(),
            metadata: HashMap::new(),
        };

        let embedding = vec![0.1; 384];
        store.store_message(message, embedding.clone()).await?;

        let results = store.search_similar(embedding, 1).await?;
        assert!(!results.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_store_messages_batch() -> Result<()> {
        let store = create_test_store().await?;

        let messages = vec![
            (
                Message {
                    id: uuid::Uuid::new_v4().to_string(),
                    role: "user".to_string(),
                    content: "First message".to_string(),
                    timestamp: Utc::now().timestamp(),
                    metadata: HashMap::new(),
                },
                vec![0.1; 384],
            ),
            (
                Message {
                    id: uuid::Uuid::new_v4().to_string(),
                    role: "assistant".to_string(),
                    content: "Second message".to_string(),
                    timestamp: Utc::now().timestamp(),
                    metadata: HashMap::new(),
                },
                vec![0.2; 384],
            ),
        ];

        store.store_messages_batch(messages.clone()).await?;

        // Verify messages were stored
        for (message, _) in messages {
            let retrieved = store.get_message(&message.id).await?;
            assert!(retrieved.is_some());
            assert_eq!(retrieved.unwrap().content, message.content);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_delete_messages_batch() -> Result<()> {
        let store = create_test_store().await?;

        let messages = vec![
            (
                Message {
                    id: uuid::Uuid::new_v4().to_string(),
                    role: "user".to_string(),
                    content: "Message to delete 1".to_string(),
                    timestamp: Utc::now().timestamp(),
                    metadata: HashMap::new(),
                },
                vec![0.1; 384],
            ),
            (
                Message {
                    id: uuid::Uuid::new_v4().to_string(),
                    role: "assistant".to_string(),
                    content: "Message to delete 2".to_string(),
                    timestamp: Utc::now().timestamp(),
                    metadata: HashMap::new(),
                },
                vec![0.2; 384],
            ),
        ];

        // Store messages
        store.store_messages_batch(messages.clone()).await?;

        // Delete messages
        let message_ids: Vec<String> = messages.iter().map(|(m, _)| m.id.clone()).collect();
        store.delete_messages_batch(message_ids.clone()).await?;

        // Verify messages were deleted
        for id in message_ids {
            let retrieved = store.get_message(&id).await?;
            assert!(retrieved.is_none());
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_search_similar_batch() -> Result<()> {
        let store = create_test_store().await?;

        // Store test messages
        let messages = vec![
            (
                Message {
                    id: uuid::Uuid::new_v4().to_string(),
                    role: "user".to_string(),
                    content: "First test message".to_string(),
                    timestamp: Utc::now().timestamp(),
                    metadata: HashMap::new(),
                },
                vec![0.1; 384],
            ),
            (
                Message {
                    id: uuid::Uuid::new_v4().to_string(),
                    role: "assistant".to_string(),
                    content: "Second test message".to_string(),
                    timestamp: Utc::now().timestamp(),
                    metadata: HashMap::new(),
                },
                vec![0.2; 384],
            ),
        ];

        store.store_messages_batch(messages.clone()).await?;

        // Test batch search
        let search_embeddings = vec![vec![0.1; 384], vec![0.2; 384]];
        let results = store.search_similar_batch(search_embeddings, 1).await?;

        assert_eq!(results.len(), 2);
        assert!(!results[0].is_empty());
        assert!(!results[1].is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_invalid_dimension() -> Result<()> {
        let store = create_test_store().await?;

        let message = Message {
            id: uuid::Uuid::new_v4().to_string(),
            role: "user".to_string(),
            content: "Test message".to_string(),
            timestamp: Utc::now().timestamp(),
            metadata: HashMap::new(),
        };

        // Test with invalid dimension
        let result = store.store_message(message, vec![0.1; 100]).await;
        assert!(result.is_err());

        // Test batch search with invalid dimension
        let result = store.search_similar_batch(vec![vec![0.1; 100]], 1).await;
        assert!(result.is_err());

        Ok(())
    }
}
 