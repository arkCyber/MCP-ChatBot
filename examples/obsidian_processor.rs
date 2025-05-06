use anyhow::Result;
use env_logger::Env;
use log::info;
use mcp_chatbot::document_processor::DocumentProcessor;
use mcp_chatbot::vector_store::VectorStore;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    // 初始化日志
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    info!("Starting Obsidian document processor...");

    // 创建模型缓存目录
    let cache_dir = Path::new("models");
    if !cache_dir.exists() {
        std::fs::create_dir_all(cache_dir)?;
    }

    // 初始化向量存储
    let vector_store = VectorStore::new("http://localhost:6333", "obsidian_docs", 384).await?;
    info!("Vector store initialized");

    // 设置 Obsidian vault 路径
    let vault_path = std::env::args()
        .nth(1)
        .expect("Please provide the path to your Obsidian vault");

    // 创建文档处理器
    let processor = DocumentProcessor::new(&vault_path, vector_store).await?;
    info!("Document processor initialized");

    // 处理整个 vault
    let documents = processor.process_vault().await?;
    info!("Processed {} documents", documents.len());

    // 示例：搜索相似文档
    let query = "人工智能";
    info!("\nSearching for documents similar to: {}", query);
    let similar_docs = processor.search_similar_documents(query, 3).await?;

    println!("\nSimilar documents found:");
    for (i, doc) in similar_docs.iter().enumerate() {
        println!("\n{}. {}", i + 1, doc.title);
        println!("Path: {}", doc.path);
        println!(
            "Content preview: {}",
            doc.content.chars().take(200).collect::<String>()
        );
    }

    Ok(())
}
