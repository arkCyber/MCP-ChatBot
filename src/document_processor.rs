use anyhow::Result;
use chrono::Utc;
use log::{debug, info};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use uuid::Uuid;
use walkdir::WalkDir;

use crate::vector_store::{Message, VectorStore};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub title: String,
    pub content: String,
    pub path: String,
    pub created_at: i64,
    pub updated_at: i64,
    pub metadata: std::collections::HashMap<String, String>,
}

pub struct DocumentProcessor {
    vector_store: VectorStore,
    model: rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel,
    vault_path: PathBuf,
}

impl DocumentProcessor {
    pub async fn new(vault_path: &str, vector_store: VectorStore) -> Result<Self> {
        let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
            .with_device(tch::Device::Cpu)
            .create_model()?;

        Ok(Self {
            vector_store,
            model,
            vault_path: PathBuf::from(vault_path),
        })
    }

    pub async fn process_vault(&self) -> Result<Vec<Document>> {
        info!(
            "Processing Obsidian vault at: {}",
            self.vault_path.display()
        );
        let mut documents = Vec::new();

        for entry in WalkDir::new(&self.vault_path)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.path().extension().map_or(false, |ext| ext == "md") {
                if let Ok(document) = self.process_document(entry.path()).await {
                    documents.push(document);
                }
            }
        }

        info!("Processed {} documents", documents.len());
        Ok(documents)
    }

    async fn process_document(&self, path: &Path) -> Result<Document> {
        let content = tokio::fs::read_to_string(path).await?;
        let title = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Untitled")
            .to_string();

        let document = Document {
            id: Uuid::new_v4().to_string(),
            title,
            content: content.clone(),
            path: path.to_string_lossy().to_string(),
            created_at: Utc::now().timestamp(),
            updated_at: Utc::now().timestamp(),
            metadata: std::collections::HashMap::new(),
        };

        // 生成文档的向量表示
        let embedding = self.model.encode(&[content])?;

        // 存储到向量数据库
        let message = Message {
            id: document.id.clone(),
            role: "document".to_string(),
            content: document.content.clone(),
            timestamp: document.created_at,
            metadata: document.metadata.clone(),
        };

        self.vector_store
            .store_message(message, embedding[0].clone())
            .await?;

        Ok(document)
    }

    pub async fn search_similar_documents(&self, query: &str, limit: u64) -> Result<Vec<Document>> {
        // 生成查询的向量表示
        let query_embedding = self.model.encode(&[query.to_string()])?;

        // 在向量数据库中搜索相似文档
        let results = self
            .vector_store
            .search_similar(query_embedding[0].clone(), limit)
            .await?;

        let mut documents = Vec::new();
        for (id, score, metadata) in results {
            let document = Document {
                id: id.clone(),
                title: metadata
                    .get("title")
                    .unwrap_or(&"Untitled".to_string())
                    .clone(),
                content: metadata.get("content").unwrap_or(&"".to_string()).clone(),
                path: metadata.get("path").unwrap_or(&"".to_string()).clone(),
                created_at: metadata
                    .get("created_at")
                    .and_then(|t| t.parse().ok())
                    .unwrap_or(0),
                updated_at: metadata
                    .get("updated_at")
                    .and_then(|t| t.parse().ok())
                    .unwrap_or(0),
                metadata,
            };
            documents.push(document);
        }

        Ok(documents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    async fn create_test_processor() -> Result<DocumentProcessor> {
        let temp_dir = tempdir()?;
        let vector_store =
            VectorStore::new("http://localhost:6333", "test_collection", 384).await?;
        DocumentProcessor::new(temp_dir.path().to_str().unwrap(), vector_store).await
    }

    #[tokio::test]
    async fn test_process_document() -> Result<()> {
        let processor = create_test_processor().await?;
        let temp_dir = tempdir()?;
        let test_file = temp_dir.path().join("test.md");

        tokio::fs::write(&test_file, "# Test Document\n\nThis is a test document.").await?;

        let document = processor.process_document(&test_file).await?;
        assert_eq!(document.title, "test");
        assert!(document.content.contains("Test Document"));

        Ok(())
    }

    #[tokio::test]
    async fn test_search_similar_documents() -> Result<()> {
        let processor = create_test_processor().await?;
        let temp_dir = tempdir()?;
        let test_file = temp_dir.path().join("test.md");

        tokio::fs::write(
            &test_file,
            "# Test Document\n\nThis is a test document about AI.",
        )
        .await?;

        processor.process_document(&test_file).await?;

        let results = processor
            .search_similar_documents("AI technology", 1)
            .await?;
        assert!(!results.is_empty());
        assert!(results[0].content.contains("AI"));

        Ok(())
    }
}
