use anyhow::Result;
use chrono::Utc;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::vector_store::{Message, VectorStore};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub created_at: i64,
    pub updated_at: i64,
    pub messages: Vec<Message>,
}

#[derive(Clone)]
pub struct ConversationManager {
    conversations: Arc<DashMap<String, Conversation>>,
    vector_store: Arc<VectorStore>,
}

impl ConversationManager {
    pub fn new(vector_store: VectorStore) -> Self {
        Self {
            conversations: Arc::new(DashMap::new()),
            vector_store: Arc::new(vector_store),
        }
    }

    pub async fn create_conversation(&self, title: &str) -> Result<Conversation> {
        let conversation = Conversation {
            id: Uuid::new_v4().to_string(),
            title: title.to_string(),
            created_at: Utc::now().timestamp(),
            updated_at: Utc::now().timestamp(),
            messages: Vec::new(),
        };

        self.conversations
            .insert(conversation.id.clone(), conversation.clone());
        Ok(conversation)
    }

    pub async fn add_message(
        &self,
        conversation_id: &str,
        role: &str,
        content: &str,
        embedding: Vec<f32>,
    ) -> Result<Message> {
        let message = Message {
            id: Uuid::new_v4().to_string(),
            role: role.to_string(),
            content: content.to_string(),
            timestamp: Utc::now().timestamp(),
            metadata: std::collections::HashMap::new(),
        };

        // 存储消息到向量数据库
        self.vector_store
            .store_message(message.clone(), embedding)
            .await?;

        // 更新对话历史
        if let Some(mut conversation) = self.conversations.get_mut(conversation_id) {
            conversation.messages.push(message.clone());
            conversation.updated_at = Utc::now().timestamp();
        }

        Ok(message)
    }

    pub async fn get_conversation(&self, conversation_id: &str) -> Option<Conversation> {
        self.conversations.get(conversation_id).map(|c| c.clone())
    }

    pub async fn list_conversations(&self) -> Vec<Conversation> {
        self.conversations.iter().map(|c| c.clone()).collect()
    }

    pub async fn delete_conversation(&self, conversation_id: &str) -> Result<()> {
        if let Some(conversation) = self.conversations.remove(conversation_id) {
            // 删除相关的向量存储
            for message in conversation.1.messages {
                self.vector_store.delete_message(&message.id).await?;
            }
        }
        Ok(())
    }

    pub async fn search_similar_messages(
        &self,
        embedding: Vec<f32>,
        limit: u64,
    ) -> Result<Vec<Message>> {
        let results = self.vector_store.search_similar(embedding, limit).await?;

        let mut messages = Vec::new();
        for (id, _score, metadata) in results {
            let message = Message {
                id: id.clone(),
                role: metadata
                    .get("role")
                    .unwrap_or(&"unknown".to_string())
                    .clone(),
                content: metadata.get("content").unwrap_or(&"".to_string()).clone(),
                timestamp: metadata
                    .get("timestamp")
                    .and_then(|t| t.parse().ok())
                    .unwrap_or(0),
                metadata,
            };
            messages.push(message);
        }

        Ok(messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    async fn create_test_manager() -> Result<ConversationManager> {
        let vector_store =
            VectorStore::new("http://localhost:6333", "test_collection", 384).await?;
        Ok(ConversationManager::new(vector_store))
    }

    #[tokio::test]
    async fn test_create_and_retrieve_conversation() -> Result<()> {
        let manager = create_test_manager().await?;

        let conversation = manager.create_conversation("Test Conversation").await?;
        let retrieved = manager.get_conversation(&conversation.id).await;

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().title, "Test Conversation");

        Ok(())
    }

    #[tokio::test]
    async fn test_add_message() -> Result<()> {
        let manager = create_test_manager().await?;

        let conversation = manager.create_conversation("Test Conversation").await?;
        let message = manager
            .add_message(&conversation.id, "user", "Hello, world!", vec![0.1; 384])
            .await?;

        let retrieved = manager.get_conversation(&conversation.id).await.unwrap();
        assert_eq!(retrieved.messages.len(), 1);
        assert_eq!(retrieved.messages[0].content, message.content);

        Ok(())
    }
}
