//! MCP-ChatBot Library: Core components for the chatbot framework
//!
//! This library provides the core functionality for the MCP-ChatBot system, including:
//! - Server implementations (file, SQLite, stdio)
//! - LLM client integration
//! - Protocol definitions
//! - Utility functions
//! - Configuration management
//!
//! The modules are organized as follows:
//! - `file_server`: File system operations and management
//! - `llm_client`: Language model client implementations
//! - `mcp_server`: Core MCP server functionality
//! - `protocol`: Communication protocol definitions
//! - `sqlite_server`: SQLite database integration
//! - `stdio_server`: Standard I/O handling
//! - `utils`: Common utility functions
//! - `config`: Configuration management
//! - `rag_server`: RAG server functionality
//!
//! Author: arkSong <arksong2018@gmail.com>
//! Version: 1.0.0
//! License: MIT

pub mod config;
pub mod conversation;
pub mod document_processor;
pub mod file_server;
pub mod llm_client;
pub mod mcp_server;
pub mod protocol;
pub mod rag_server;
pub mod sqlite_server;
pub mod stdio_server;
pub mod utils;
pub mod vector_db;
pub mod vector_store;

// Re-export commonly used types
pub use config::Configuration;
pub use llm_client::{LLMClient, LLMConfig, LLMProvider};
pub use mcp_server::{McpServer, Resource, Tool};
pub use protocol::{Message, ToolSchema};
pub use sqlite_server::SqliteServer;
pub use vector_db::*;

pub struct ChatSession {
    pub llm_client: std::sync::Arc<tokio::sync::Mutex<LLMClient>>,
    pub servers: Vec<Server>,
    pub current_provider: LLMProvider,
    pub running: bool,
}

impl ChatSession {
    pub async fn new(
        llm_client: std::sync::Arc<tokio::sync::Mutex<LLMClient>>,
    ) -> anyhow::Result<Self> {
        let mut servers = Vec::new();
        let config = Configuration::new();
        let llm_config = config.load_llm_config("config/config.yaml")?;

        let server_config = std::fs::read_to_string("config/servers_config.json")?;
        let server_config: serde_json::Value = serde_json::from_str(&server_config)?;

        if let Some(servers_config) = server_config.get("mcpServers") {
            for (name, config) in servers_config.as_object().unwrap() {
                let server = Server::new(name.clone(), config.clone()).await?;
                servers.push(server);
            }
        }

        Ok(Self {
            llm_client,
            servers,
            current_provider: LLMProvider::Ollama,
            running: true,
        })
    }

    pub async fn process_command(&mut self, command: &str) -> anyhow::Result<()> {
        match command {
            "/help" => Ok(()),
            "/clear" => Ok(()),
            "/exit" => {
                self.running = false;
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

pub struct Server {
    pub name: String,
    pub config: serde_json::Value,
    pub mcp_server: McpServer,
}

impl Server {
    pub async fn new(name: String, config: serde_json::Value) -> anyhow::Result<Self> {
        let mut mcp_server = McpServer::new();
        mcp_server.initialize().await?;

        Ok(Self {
            name,
            config,
            mcp_server,
        })
    }

    pub async fn execute_tool(
        &mut self,
        tool_name: &str,
        arguments: &serde_json::Value,
        retries: u32,
        delay: std::time::Duration,
    ) -> anyhow::Result<serde_json::Value> {
        self.mcp_server
            .execute_tool(tool_name, arguments, retries, delay)
            .await
    }
}

// Re-export the load_system_prompts function
pub fn load_system_prompts() -> SystemPrompts {
    match std::fs::read_to_string("mcp_prompts.yaml") {
        Ok(content) => match serde_yaml::from_str::<SystemPrompts>(&content) {
            Ok(prompts) => prompts,
            Err(_) => create_default_prompts(),
        },
        Err(_) => create_default_prompts(),
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct SystemPrompts {
    pub server_prompts: std::collections::HashMap<String, ServerPrompt>,
    pub default_system_prompt: String,
    pub common_prompts: CommonPrompts,
}

#[derive(Debug, serde::Deserialize)]
pub struct ServerPrompt {
    pub system_prompt: String,
}

#[derive(Debug, serde::Deserialize)]
pub struct CommonPrompts {
    pub tool_response: String,
    pub welcome: String,
}

fn create_default_prompts() -> SystemPrompts {
    SystemPrompts {
        server_prompts: std::collections::HashMap::new(),
        default_system_prompt: "You are an intelligent assistant.".to_string(),
        common_prompts: CommonPrompts {
            tool_response: "Tool response received.".to_string(),
            welcome: "Welcome to MCP-ChatBot!".to_string(),
        },
    }
}
