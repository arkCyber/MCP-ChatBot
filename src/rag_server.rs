//! RAG Server Module: Retrieval Augmented Generation using Qdrant
//!
//! This module provides functionality for:
//! - Document embedding and storage
//! - Semantic search using vector similarity
//! - Context augmentation for LLM queries
//!
//! Author: arkSong <arksong2018@gmail.com>
//! Version: 1.0.0
//! License: MIT

use anyhow::{Context, Error, Result};
use log::{debug, info};
use lru::LruCache;
use ndarray::{Array1, Array2};
use ndarray_linalg::Norm;
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};
use qdrant_client::qdrant::{
    point_id, r#match::MatchValue, value::Kind, vectors, vectors_config::Config, CreateCollection,
    Datatype, Distance, Filter, OptimizersConfigDiff, PointId, PointStruct, SearchPoints,
    SearchPointsBuilder, UpsertPointsBuilder, Value, Vector, VectorParams, Vectors, VectorsConfig,
    WalConfigDiff,
};
use qdrant_client::Qdrant;
use regex::Regex;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsConfig, SentenceEmbeddingsModel,
    SentenceEmbeddingsModelType,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use tokenizers::tokenizer::TruncationDirection;
use tokenizers::Tokenizer;
use tokio::sync::RwLock as TokioRwLock;
use unicode_normalization::UnicodeNormalization;
use uuid;

const EMBEDDING_SIZE: usize = 768;
const MAX_CACHE_SIZE: usize = 10000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: String,
    pub document_id: String,
    pub text: String,
    pub chunk_index: usize,
    pub metadata: Option<serde_json::Value>,
}

pub struct RagServer {
    client: Qdrant,
    model: SentenceEmbeddingsModel,
    text_cleaner: Arc<Regex>,
    chunk_size: usize,
    chunk_overlap: usize,
    embedding_cache: Arc<TokioRwLock<LruCache<String, Vec<f32>>>>,
    collection_name: String,
}

impl RagServer {
    pub fn new() -> Result<Self> {
        let client = Qdrant::from_url("http://localhost:6334").build()?;
        let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
            .with_device(tch::Device::Cpu)
            .create_model()?;
        let text_cleaner = Regex::new(r"[^\p{L}\p{N}\s]")?;
        let cache = LruCache::new(NonZeroUsize::new(MAX_CACHE_SIZE).unwrap());
        Ok(Self {
            client,
            model,
            text_cleaner: Arc::new(text_cleaner),
            chunk_size: 512,
            chunk_overlap: 128,
            embedding_cache: Arc::new(TokioRwLock::new(cache)),
            collection_name: "documents".to_string(),
        })
    }

    pub async fn init(&self) -> Result<()> {
        let collections = self.client.list_collections().await?;
        let collection_exists = collections
            .collections
            .iter()
            .any(|c| c.name == self.collection_name);

        if !collection_exists {
            info!("Creating collection: {}", self.collection_name);
            self.create_collection().await?;
        } else {
            info!("Collection {} already exists", self.collection_name);
        }

        Ok(())
    }

    async fn create_collection(&self) -> Result<()> {
        let vector_config = VectorParams {
            size: EMBEDDING_SIZE as u64,
            distance: Distance::Cosine.into(),
            on_disk: Some(true),
            ..Default::default()
        };

        self.client
            .create_collection(
                qdrant_client::qdrant::CreateCollectionBuilder::new(&self.collection_name)
                    .vectors_config(VectorsConfig {
                        config: Some(Config::Params(vector_config)),
                    })
                    .optimizers_config(OptimizersConfigDiff {
                        indexing_threshold: Some(20000),
                        memmap_threshold: Some(50000),
                        ..Default::default()
                    })
                    .wal_config(WalConfigDiff {
                        wal_capacity_mb: Some(32),
                        wal_segments_ahead: Some(64),
                        ..Default::default()
                    })
                    .on_disk_payload(true),
            )
            .await?;

        Ok(())
    }

    pub fn preprocess_text(&self, text: &str) -> String {
        let cleaned = self.text_cleaner.replace_all(text, " ");
        cleaned
            .to_lowercase()
            .nfkd()
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn split_into_chunks(&self, text: &str) -> Vec<String> {
        // Split text into paragraphs first
        let paragraphs: Vec<&str> = text.split("\n\n").collect();
        let mut chunks = Vec::new();

        for paragraph in paragraphs {
            // Split paragraph into sentences
            let sentences: Vec<&str> = paragraph
                .split(&['.', '!', '?'])
                .filter(|s| !s.trim().is_empty())
                .collect();

            let mut current_chunk = String::new();
            let mut current_size = 0;

            for sentence in sentences {
                let sentence = sentence.trim();
                let sentence_words: Vec<&str> = sentence.split_whitespace().collect();

                if current_size + sentence_words.len() > self.chunk_size {
                    if !current_chunk.is_empty() {
                        chunks.push(current_chunk.trim().to_string());
                        current_chunk = String::new();
                        current_size = 0;
                    }

                    // If a single sentence is longer than chunk_size, split it
                    if sentence_words.len() > self.chunk_size {
                        let mut start = 0;
                        while start < sentence_words.len() {
                            let end = std::cmp::min(start + self.chunk_size, sentence_words.len());
                            chunks.push(sentence_words[start..end].join(" "));
                            start = end.saturating_sub(self.chunk_overlap);
                        }
                    } else {
                        current_chunk = sentence.to_string();
                        current_size = sentence_words.len();
                    }
                } else {
                    if !current_chunk.is_empty() {
                        current_chunk.push(' ');
                    }
                    current_chunk.push_str(sentence);
                    current_size += sentence_words.len();
                }
            }

            if !current_chunk.is_empty() {
                chunks.push(current_chunk.trim().to_string());
            }
        }

        chunks
    }

    pub async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        let cached_embedding = {
            let cache = self.embedding_cache.read().await;
            cache.peek(text).map(Vec::clone)
        };

        if let Some(embedding) = cached_embedding {
            return Ok(embedding);
        }

        // Generate embedding using rust-bert
        let embeddings = self.model.encode(&[text])?;
        let embedding = embeddings[0].clone();

        // Cache the result
        {
            let mut cache = self.embedding_cache.write().await;
            cache.put(text.to_string(), embedding.clone());
        }

        Ok(embedding)
    }

    pub async fn add_document(&mut self, doc: Document) -> Result<()> {
        let chunks = self.split_into_chunks(&doc.text);

        for (i, chunk_text) in chunks.iter().enumerate() {
            let embedding = self.generate_embedding(chunk_text).await?;

            let point = PointStruct {
                id: Some(PointId::from(uuid::Uuid::new_v4().to_string())),
                vectors: Some(Vectors::from(embedding)),
                payload: {
                    let mut payload = HashMap::new();
                    payload.insert("text".to_string(), Value::from(chunk_text.as_str()));
                    payload.insert("document_id".to_string(), Value::from(doc.id.as_str()));
                    payload.insert(
                        "chunk_index".to_string(),
                        Value::from(i.to_string().as_str()),
                    );

                    // Fix metadata serialization
                    if let Some(metadata) = &doc.metadata {
                        payload.insert(
                            "metadata".to_string(),
                            Value::from(serde_json::to_string(metadata)?.as_str()),
                        );
                    }

                    payload
                },
            };

            self.client
                .upsert_points(
                    UpsertPointsBuilder::new(&self.collection_name, vec![point])
                        .wait(true)
                        .build(),
                )
                .await?;
        }

        Ok(())
    }

    pub async fn search_similar(
        &self,
        query: &str,
        limit: u32,
        filter: Option<HashMap<String, Value>>,
    ) -> Result<Vec<SearchResult>> {
        let query_embedding = self.generate_embedding(query).await?;

        let mut search_builder = SearchPointsBuilder::new(
            &self.collection_name,
            query_embedding.to_vec(),
            limit as u64,
        )
        .with_payload(true);

        if let Some(filter_map) = filter {
            let conditions: Vec<qdrant_client::qdrant::Condition> = filter_map
                .into_iter()
                .map(|(k, v)| {
                    let field_condition = qdrant_client::qdrant::FieldCondition {
                        key: k,
                        r#match: Some(qdrant_client::qdrant::Match {
                            match_value: Some(MatchValue::Keyword(
                                v.as_str().map_or("".to_string(), |s| s.to_string()),
                            )),
                        }),
                        range: None,
                        geo_radius: None,
                        values_count: None,
                        geo_bounding_box: None,
                        geo_polygon: None,
                        datetime_range: None,
                        is_empty: None,
                        is_null: None,
                    };
                    qdrant_client::qdrant::Condition {
                        condition_one_of: Some(
                            qdrant_client::qdrant::condition::ConditionOneOf::Field(
                                field_condition,
                            ),
                        ),
                    }
                })
                .collect();

            let filter = Filter {
                should: Vec::new(),
                must: conditions,
                must_not: Vec::new(),
                min_should: None,
            };
            search_builder = search_builder.filter(filter);
        }

        let search_result = self.client.search_points(search_builder.build()).await?;

        let results = search_result
            .result
            .into_iter()
            .filter_map(|scored_point| {
                let score = scored_point.score;
                let payload = scored_point.payload;

                let text = payload.get("text").and_then(|v| match v {
                    Value {
                        kind: Some(Kind::StringValue(s)),
                    } => Some(s.clone()),
                    _ => None,
                })?;

                let document_id = payload.get("document_id").and_then(|v| match v {
                    Value {
                        kind: Some(Kind::StringValue(s)),
                    } => Some(s.clone()),
                    _ => None,
                })?;

                let chunk_index = payload.get("chunk_index").and_then(|v| match v {
                    Value {
                        kind: Some(Kind::StringValue(s)),
                    } => s.parse::<usize>().ok(),
                    _ => None,
                })?;

                let metadata = payload.get("metadata").and_then(|v| match v {
                    Value {
                        kind: Some(Kind::StringValue(s)),
                    } => serde_json::from_str(s).ok(),
                    _ => None,
                });

                Some(SearchResult {
                    text,
                    document_id,
                    chunk_index,
                    score,
                    metadata,
                })
            })
            .collect();

        Ok(results)
    }

    pub async fn get_collection_info(&self) -> Result<HashMap<String, String>> {
        let info = self.client.collection_info(&self.collection_name).await?;

        let mut result = HashMap::new();
        if let Some(result_info) = info.result {
            result.insert("name".to_string(), self.collection_name.clone());
            if let Some(config) = result_info.config {
                if let Some(params) = config.params {
                    if let Some(vector_config) = params.vectors_config {
                        if let Some(vector_params) = vector_config.config {
                            if let Config::Params(params) = vector_params {
                                result.insert("vector_size".to_string(), params.size.to_string());
                                result.insert(
                                    "distance".to_string(),
                                    format!("{:?}", params.distance),
                                );
                            }
                        }
                    }
                }
            }
            if let Some(points_count) = result_info.points_count {
                result.insert("points_count".to_string(), points_count.to_string());
            }
        }

        Ok(result)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub text: String,
    pub document_id: String,
    pub chunk_index: usize,
    pub score: f32,
    pub metadata: Option<serde_json::Value>,
}
