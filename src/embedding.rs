use anyhow::{Context, Result};
use log::{debug, info, warn};
use rayon::prelude::*;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct EmbeddingModel {
    model: Arc<Mutex<rust_bert::pipelines::sentence_embeddings::SentenceEmbeddings>>,
    batch_size: usize,
    device: tch::Device,
}

impl EmbeddingModel {
    pub fn new() -> Result<Self> {
        info!("Initializing Sentence-Transformers model...");

        // 检查并创建模型缓存目录
        let cache_dir = Path::new("models");
        if !cache_dir.exists() {
            std::fs::create_dir_all(cache_dir).context("Failed to create model cache directory")?;
        }

        // 使用 M1 的 MPS 设备
        let device = if tch::utils::has_mps() {
            info!("Using M1 MPS device");
            tch::Device::Mps
        } else {
            info!("MPS not available, falling back to CPU");
            tch::Device::Cpu
        };

        // 使用 all-MiniLM-L6-v2 模型
        let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
            .with_device(device)
            .with_cache_dir(Some(cache_dir.to_path_buf()))
            .create_model()
            .context("Failed to create embedding model")?;

        info!("Model initialized successfully");

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            batch_size: 64, // M1 内存较大，增加批处理大小
            device,
        })
    }

    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let model = self.model.lock().await;
        let embeddings = model
            .encode(&[text])
            .context("Failed to generate embeddings")?;

        Ok(embeddings[0].to_vec())
    }

    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let model = self.model.lock().await;

        // 分批处理以避免内存问题
        let mut all_embeddings = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            let embeddings = model
                .encode(chunk)
                .context("Failed to generate batch embeddings")?;
            all_embeddings.extend(embeddings.into_iter().map(|v| v.to_vec()));
        }

        Ok(all_embeddings)
    }

    pub async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f32>>> {
        info!("Embedding {} documents", documents.len());

        // 使用 rayon 进行并行处理
        let embeddings = tokio::task::spawn_blocking(move || {
            documents
                .par_chunks(self.batch_size)
                .flat_map(|chunk| {
                    let model = self.model.lock().await;
                    model.encode(chunk).unwrap_or_else(|e| {
                        warn!("Failed to encode chunk: {}", e);
                        vec![]
                    })
                })
                .map(|v| v.to_vec())
                .collect::<Vec<_>>()
        })
        .await
        .context("Failed to process documents")?;

        info!("Successfully embedded {} documents", embeddings.len());
        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embedding() {
        let model = EmbeddingModel::new().unwrap();

        // 测试单个文本
        let text = "This is a test sentence.";
        let embedding = model.embed_text(text).await.unwrap();
        assert_eq!(embedding.len(), 384); // all-MiniLM-L6-v2 输出维度为 384

        // 测试批量文本
        let texts = vec![
            "First test sentence.".to_string(),
            "Second test sentence.".to_string(),
        ];
        let embeddings = model.embed_batch(&texts).await.unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
    }
}
