use anyhow::Result;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize the model
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
        .with_device(tch::Device::Cpu)
        .create_model()?;

    // Example text
    let text = "This is a test sentence for embedding generation.";

    // Generate embedding
    let embeddings = model.encode(&[text])?;
    let embedding = &embeddings[0];

    println!("Generated embedding: {:?}", embedding);
    println!("Embedding dimension: {}", embedding.len());

    Ok(())
}

// 计算余弦相似度
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}
