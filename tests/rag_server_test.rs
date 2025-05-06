use anyhow::Result;
use log::{debug, info};
use mcp_chatbot::rag_server::{Document, RagServer};
use qdrant_client::qdrant::Value;
use std::collections::HashMap;

#[tokio::test]
async fn test_text_preprocessing() -> Result<()> {
    info!("=== Starting text preprocessing test ===");
    info!("Step 1/3: Creating RAG server instance...");
    let server = RagServer::new()?;
    info!("Step 2/3: Testing text preprocessing...");
    let text = "Hello, World! 123";
    info!("Input text: {}", text);
    let cleaned = server.preprocess_text(text);
    info!("Cleaned text: {}", cleaned);
    info!("Step 3/3: Verifying results...");
    assert_eq!(cleaned, "hello world 123");
    info!("=== Text preprocessing test completed successfully ===\n");
    Ok(())
}

#[tokio::test]
async fn test_rag_server() -> Result<()> {
    info!("=== Starting RAG server test ===");

    info!("Step 1/5: Creating RAG server instance...");
    let mut server = RagServer::new()?;
    info!("Step 2/5: Initializing RAG server...");
    server.init().await?;
    info!("RAG server initialized successfully");

    info!("Step 3/5: Creating test document...");
    let doc = Document {
        id: "test1".to_string(),
        text: "This is a test document".to_string(),
        metadata: None,
    };
    info!("Test document created with id: {}", doc.id);

    info!("Step 4/5: Adding document to RAG server...");
    server.add_document(doc).await?;
    info!("Document added successfully");

    info!("Step 5/5: Performing similarity search...");
    let results = server.search_similar("test document", 5, None).await?;
    info!("Search completed, found {} results", results.len());

    assert!(!results.is_empty());
    info!("=== RAG server test completed successfully ===\n");
    Ok(())
}

#[tokio::test]
async fn test_document_chunking() -> Result<()> {
    info!("=== Starting document chunking test ===");
    info!("Step 1/4: Creating RAG server instance...");
    let server = RagServer::new()?;
    info!("Step 2/4: Preparing test text...");
    let text = "This is a test document. It has multiple sentences. We want to test how it gets chunked. This is a longer sentence that might need to be split into multiple chunks because it contains more words than the chunk size limit.";
    info!("Input text length: {} characters", text.len());

    info!("Step 3/4: Splitting text into chunks...");
    let chunks = server.split_into_chunks(text);
    info!("Text split into {} chunks", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        debug!("Chunk {} content: {}", i + 1, chunk);
        info!("Chunk {}: {} characters", i + 1, chunk.len());
    }

    info!("Step 4/4: Verifying results...");
    assert!(!chunks.is_empty());
    info!("=== Document chunking test completed successfully ===\n");
    Ok(())
}

#[tokio::test]
async fn test_embedding_generation() -> Result<()> {
    info!("=== Starting embedding generation test ===");
    info!("Step 1/4: Creating RAG server instance...");
    let server = RagServer::new()?;
    info!("Step 2/4: Preparing test text...");
    let text = "This is a test for embedding generation";
    info!("Input text: {}", text);

    info!("Step 3/4: Generating embedding...");
    let embedding = server.generate_embedding(text).await?;
    info!("Generated embedding with size: {}", embedding.len());
    debug!("First few embedding values: {:?}", &embedding[..5]);

    info!("Step 4/4: Verifying results...");
    assert_eq!(embedding.len(), 768); // EMBEDDING_SIZE
    info!("=== Embedding generation test completed successfully ===\n");
    Ok(())
}

#[tokio::test]
async fn test_collection_info() -> Result<()> {
    info!("=== Starting collection info test ===");
    info!("Step 1/4: Creating RAG server instance...");
    let server = RagServer::new()?;
    info!("Step 2/4: Initializing RAG server...");
    server.init().await?;

    info!("Step 3/4: Retrieving collection info...");
    let info = server.get_collection_info().await?;
    info!("Collection info retrieved: {:?}", info);
    debug!("Detailed collection info: {:#?}", info);

    info!("Step 4/4: Verifying results...");
    assert!(!info.is_empty());
    assert!(info.contains_key("name"));
    assert!(info.contains_key("vector_size"));
    assert!(info.contains_key("distance"));
    assert!(info.contains_key("points_count"));

    // Verify specific values
    assert_eq!(info["name"], "documents");
    assert_eq!(info["vector_size"], "768");
    assert_eq!(info["distance"], "Cosine");
    assert_eq!(info["points_count"], "0"); // Collection should be empty initially

    info!("=== Collection info test completed successfully ===\n");
    Ok(())
}

#[tokio::test]
async fn test_document_operations() -> Result<()> {
    info!("=== Starting document operations test ===");
    info!("Step 1/5: Creating RAG server instance...");
    let mut server = RagServer::new()?;
    info!("Step 2/5: Initializing RAG server...");
    server.init().await?;

    info!("Step 3/5: Creating test document...");
    let doc = Document {
        id: "test2".to_string(),
        text: "Another test document for operations".to_string(),
        metadata: Some(serde_json::json!({
            "source": "test",
            "timestamp": "2024-03-20"
        })),
    };
    info!("Adding document with id: {}", doc.id);
    server.add_document(doc).await?;
    info!("Document added successfully");

    info!("Step 4/5: Searching with metadata filter...");
    let filter = Some(HashMap::from([("source".to_string(), Value::from("test"))]));
    let results = server.search_similar("test document", 5, filter).await?;
    info!("Search completed, found {} results", results.len());
    debug!("Search results: {:#?}", results);

    info!("Step 5/5: Verifying results...");
    assert!(!results.is_empty());
    info!("=== Document operations test completed successfully ===\n");
    Ok(())
}

#[tokio::test]
async fn test_rag_server_basic_operations() -> Result<()> {
    // Initialize RAG server
    let mut rag_server = RagServer::new()?;
    rag_server.init().await?;

    // Test document preprocessing
    let test_text = "Hello, World! This is a test document.";
    let preprocessed = rag_server.preprocess_text(test_text);
    assert!(!preprocessed.contains("!"));
    assert!(preprocessed.contains("hello world"));

    // Test text chunking
    let long_text = "First sentence. Second sentence. Third sentence.";
    let chunks = rag_server.split_into_chunks(long_text);
    assert!(!chunks.is_empty());
    assert!(chunks.iter().any(|c| c.contains("First sentence")));
    assert!(chunks.iter().any(|c| c.contains("Second sentence")));

    // Test embedding generation
    let embedding = rag_server
        .generate_embedding("Test text for embedding")
        .await?;
    assert_eq!(embedding.len(), 768); // All-MiniLM-L6-v2 produces 768-dimensional embeddings

    // Test document addition and search
    let doc = Document {
        id: "test_doc_1".to_string(),
        text: "This is a test document about artificial intelligence and machine learning."
            .to_string(),
        metadata: Some(serde_json::json!({
            "source": "test",
            "category": "technology"
        })),
    };

    rag_server.add_document(doc).await?;

    // Test similarity search
    let results = rag_server
        .search_similar("artificial intelligence", 5, None)
        .await?;
    assert!(!results.is_empty());
    assert!(results[0].text.contains("artificial intelligence"));

    // Test search with metadata filter
    let mut filter = HashMap::new();
    filter.insert(
        "metadata".to_string(),
        serde_json::Value::String("test".to_string()),
    );
    let filtered_results = rag_server
        .search_similar("machine learning", 5, Some(filter))
        .await?;
    assert!(!filtered_results.is_empty());

    // Test collection info
    let info = rag_server.get_collection_info().await?;
    assert!(info.contains_key("name"));
    assert!(info.contains_key("vector_size"));
    assert!(info.contains_key("points_count"));

    Ok(())
}

#[tokio::test]
async fn test_rag_server_edge_cases() -> Result<()> {
    let mut rag_server = RagServer::new()?;
    rag_server.init().await?;

    // Test empty document
    let empty_doc = Document {
        id: "empty_doc".to_string(),
        text: "".to_string(),
        metadata: None,
    };
    rag_server.add_document(empty_doc).await?;

    // Test very long document
    let long_text = "This is a very long document. ".repeat(1000);
    let long_doc = Document {
        id: "long_doc".to_string(),
        text: long_text,
        metadata: None,
    };
    rag_server.add_document(long_doc).await?;

    // Test special characters
    let special_chars_doc = Document {
        id: "special_chars".to_string(),
        text: "!@#$%^&*()_+{}|:<>?~`-=[]\\;',./".to_string(),
        metadata: None,
    };
    rag_server.add_document(special_chars_doc).await?;

    // Test search with non-existent query
    let results = rag_server
        .search_similar("nonexistentquery123456", 5, None)
        .await?;
    assert!(results.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_rag_server_metadata_handling() -> Result<()> {
    let mut rag_server = RagServer::new()?;
    rag_server.init().await?;

    // Test document with complex metadata
    let complex_metadata = serde_json::json!({
        "author": "Test Author",
        "date": "2024-03-20",
        "tags": ["test", "rag", "metadata"],
        "nested": {
            "field1": "value1",
            "field2": 42
        }
    });

    let doc = Document {
        id: "metadata_test".to_string(),
        text: "Test document with complex metadata".to_string(),
        metadata: Some(complex_metadata),
    };

    rag_server.add_document(doc).await?;

    // Test search with metadata filter
    let mut filter = HashMap::new();
    filter.insert(
        "metadata".to_string(),
        serde_json::Value::String("Test Author".to_string()),
    );

    let results = rag_server
        .search_similar("test document", 5, Some(filter))
        .await?;
    assert!(!results.is_empty());

    Ok(())
}
