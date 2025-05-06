//! MCP-ChatBot: A Rust-based chatbot framework with tool integration
//!
//! This module implements a sophisticated chatbot system that combines:
//! - Language Model (LLM) integration with support for multiple providers (Ollama, OpenAI)
//! - Tool execution framework for performing various operations
//! - Server management for handling different types of services
//! - Interactive command-line interface with history and command processing
//!
//! Key Components:
//! - `Configuration`: Manages API keys and server configurations
//! - `Server`: Handles different types of servers (memory, SQLite, file system, etc.)
//! - `ChatSession`: Manages the chat interaction flow and tool execution
//! - `Message`: Represents chat messages with role and content
//!
//! Features:
//! - Multi-server support with dynamic tool registration
//! - Tool execution with retry mechanism
//! - Command-line interface with history and auto-completion
//! - Support for both Ollama and OpenAI LLM backends
//! - Resource management and cleanup
//! - Debug logging and error handling
//!
//! Usage:
//! 1. Configure servers in `config/servers_config.json`
//! 2. Set up environment variables (LLM_API_KEY, OPENAI_API_KEY)
//! 3. Run the application to start the chat session
//! 4. Use commands like /help, /tools, /servers to interact with the system
//!
//! Using Ollama Local AI Engine:
//! 1. Install Ollama from https://ollama.ai/
//! 2. Pull the required model: `ollama pull llama3.2:latest`
//! 3. Start Ollama service: `ollama serve`
//! 4. The system will automatically detect and use Ollama if available
//! 5. Use `/ai` command to switch between Ollama and OpenAI
//!
//! Example:
//! ```bash
//! # Using Ollama
//! ollama pull llama3.2:latest
//! ollama serve
//! cargo run
//!
//! # Using OpenAI
//! export OPENAI_API_KEY=your_key
//! cargo run
//! ```
//!
//! Author: arkSong <arksong2018@gmail.com>
//! Version: 1.0.0
//! License: MIT

use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Error, Result};
use dotenv::dotenv;
use log::{debug, error, info, warn};
use rustyline::config::Configurer;
use rustyline::history::FileHistory;
use rustyline::Editor;
use serde_json::{json, Value};
use serde_yaml;
use tokio::sync::Mutex;
use tokio::time::sleep;
use uuid;

mod doc_processor;
mod history;
mod llm_client;
mod mcp_server;
mod protocol;
mod rag_server;
mod sqlite_server;
mod stdio_server;
mod utils;
mod vector_db;
mod whisper_server;

use doc_processor::{DocProcessor, ObsidianConfig};
use history::History;
use llm_client::{LLMClient, LLMProvider};
use mcp_server::{McpServer, Resource, Tool};
use protocol::ToolSchema;
use rag_server::{Document, RagServer};
use utils::{
    check_ollama_status, get_server_status, print_about, print_bot_thinking_continuous,
    print_colored_ln, print_mcp_servers, print_menu, print_resources, print_tools,
    typing_animation, Color,
};
use vector_db::{VectorDBClient, VectorDBConfig};
use whisper_server::WhisperServer;

/// Default system prompts used when YAML file is not available
const DEFAULT_SYSTEM_PROMPT: &str = r#"You are an intelligent assistant that can perform various tasks. When you need to perform specific operations, you must use tools.

System Rules:
1. You must use tools when performing operations
2. Tool usage format: {"tool": "tool_name", "arguments": {"parameter_name": "value"}}
3. When using tools, only return the tool call format, do not add any explanatory text
4. If unsure which tool to use, first use the /tools command to view all available tools
5. For normal conversation, respond directly without using tools
6. Keep your responses concise and clear
7. If you don't understand something, ask for clarification
8. If you need more information, ask for it
9. If you can't perform a task, explain why"#;

const DEFAULT_TOOL_RESPONSE_PROMPT: &str = "You are a helpful assistant. Please process the following tool response and provide a clear, natural language explanation of the result. Do not include any technical details or JSON formatting in your response.";

const DEFAULT_WELCOME_MESSAGE: &str =
    "Welcome to MCP-ChatBot Playground!\nYour AI Assistant is ready to help.";

/// Represents a message in the chat system
/// Contains the role (e.g., "user", "assistant") and content of the message
#[derive(Debug, serde::Serialize)]
struct Message {
    role: String,    // The role of the message sender
    content: String, // The actual message content
}

/// Configuration structure for managing API keys and settings
struct Configuration {
    api_key: Option<String>, // Optional API key for LLM services
}

impl Configuration {
    /// Creates a new Configuration instance
    /// Loads environment variables from .env file
    fn new() -> Self {
        dotenv().ok(); // Load .env file if it exists
        Self {
            api_key: env::var("LLM_API_KEY").ok(), // Get API key from environment
        }
    }

    /// Loads configuration from a JSON file
    /// Returns a serde_json::Value containing the configuration
    fn load_config(&self, file_path: &str) -> Result<Value> {
        let contents = fs::read_to_string(file_path)?;
        Ok(serde_json::from_str(&contents)?)
    }

    /// Loads LLM configuration from a YAML file
    fn load_llm_config(&self, file_path: &str) -> Result<llm_client::LLMConfig> {
        let contents = fs::read_to_string(file_path)?;
        Ok(serde_yaml::from_str(&contents)?)
    }

    /// Determines and returns the appropriate LLM API key
    /// First checks Ollama availability, falls back to Deepseek if needed
    async fn llm_api_key(&self) -> Result<String> {
        // Test connection to Ollama server
        info!("Testing Ollama connection at http://localhost:11434/api/tags");
        let test_client = reqwest::Client::builder().no_proxy().build().unwrap();
        let test_url = "http://localhost:11434/api/tags";

        match test_client.get(test_url).send().await {
            Ok(response) => {
                info!(
                    "Received response from Ollama server: {:?}",
                    response.status()
                );
                if response.status().is_success() {
                    match response.text().await {
                        Ok(body) => {
                            info!("Ollama server response body: {}", body);
                            if body.contains("llama3.2:latest") {
                                info!("Using local Ollama server with llama3.2:latest model");
                                Ok("ollama".to_string())
                            } else {
                                error!("llama3.2:latest model not found");
                                print_colored_ln(
                                    "\nOllama model not found, please check:",
                                    Color::Red,
                                );
                                print_colored_ln("1. Make sure llama3.2:latest model is downloaded: ollama pull llama3.2:latest", Color::Yellow);
                                print_colored_ln(
                                    "\nSwitch to default Deepseek API? (y/n): ",
                                    Color::Cyan,
                                );

                                let mut input = String::new();
                                if let Ok(_) = std::io::stdin().read_line(&mut input) {
                                    if input.trim().to_lowercase() == "y" {
                                        info!("Switching to default Deepseek API");
                                        Ok("sk-878a5319c7b14bc48109e19315361".to_string())
                                    } else {
                                        Err(Error::msg("llama3.2:latest model is not available"))
                                    }
                                } else {
                                    Err(Error::msg("Failed to read user input"))
                                }
                            }
                        }
                        Err(e) => {
                            error!("Failed to read Ollama response: {}", e);
                            Err(Error::msg("Failed to read Ollama response"))
                        }
                    }
                } else {
                    error!("Ollama server returned error status: {}", response.status());
                    print_colored_ln(
                        "\nOllama server connection failed, please check:",
                        Color::Red,
                    );
                    print_colored_ln("1. Make sure Ollama service is running", Color::Yellow);
                    print_colored_ln(
                        "2. Check if Ollama service is running at http://localhost:11434",
                        Color::Yellow,
                    );
                    print_colored_ln(
                        "3. Make sure llama3.2:latest model is downloaded: ollama pull llama3.2:latest",
                        Color::Yellow,
                    );
                    print_colored_ln("\nSwitch to default Deepseek API? (y/n): ", Color::Cyan);

                    let mut input = String::new();
                    if let Ok(_) = std::io::stdin().read_line(&mut input) {
                        if input.trim().to_lowercase() == "y" {
                            info!("Switching to default Deepseek API");
                            Ok("sk-878a5319c7b14bc48109e19315361".to_string())
                        } else {
                            Err(Error::msg("Ollama server is not available"))
                        }
                    } else {
                        Err(Error::msg("Failed to read user input"))
                    }
                }
            }
            Err(e) => {
                error!("Failed to connect to Ollama server: {}", e);
                print_colored_ln(
                    "\nOllama server connection failed, please check:",
                    Color::Red,
                );
                print_colored_ln("1. Make sure Ollama service is running", Color::Yellow);
                print_colored_ln(
                    "2. Check if Ollama service is running at http://localhost:11434",
                    Color::Yellow,
                );
                print_colored_ln(
                    "3. Make sure llama3.2:latest model is downloaded: ollama pull llama3.2:latest",
                    Color::Yellow,
                );
                print_colored_ln("\nSwitch to default Deepseek API? (y/n): ", Color::Cyan);

                let mut input = String::new();
                if let Ok(_) = std::io::stdin().read_line(&mut input) {
                    if input.trim().to_lowercase() == "y" {
                        info!("Switching to default Deepseek API");
                        Ok("sk-878a5319c7b14bc48109e19315361".to_string())
                    } else {
                        Err(Error::msg("Ollama server is not available"))
                    }
                } else {
                    Err(Error::msg("Failed to read user input"))
                }
            }
        }
    }
}

/// Represents a server instance in the system
/// Manages server configuration, tools, and resources
struct Server {
    name: String,                                              // Server name/identifier
    config: Value,                                             // Server configuration
    mcp_server: McpServer,                                     // MCP server instance
    cleanup_lock: Arc<Mutex<()>>,                              // Lock for cleanup operations
    memory_store: Option<Arc<Mutex<HashMap<String, String>>>>, // Optional in-memory storage
}

impl Server {
    /// Creates a new server instance with the given name and configuration
    /// Initializes the server and registers appropriate tools and resources
    async fn new(name: String, config: Value) -> Result<Self> {
        info!("Creating new server: {}", name);

        // Get server configuration
        let command = config["command"].as_str().ok_or_else(|| {
            Error::msg(format!(
                "Missing command in server configuration for {}",
                name
            ))
        })?;

        let _args = config["args"].as_array().ok_or_else(|| {
            Error::msg(format!("Missing args in server configuration for {}", name))
        })?;

        let _env: HashMap<String, String> = config["env"]
            .as_object()
            .map(|env| {
                env.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                    .collect()
            })
            .unwrap_or_default();

        info!("Initializing server {} with command: {}", name, command);

        let mut mcp_server = McpServer::new();
        mcp_server.initialize().await?;

        // Register tools and resources based on server type
        let mut server = Self {
            name: name.clone(),
            config: config.clone(),
            mcp_server,
            cleanup_lock: Arc::new(Mutex::new(())),
            memory_store: None,
        };

        match name.as_str() {
            "memory" => {
                // Create a shared memory store
                let memory_store =
                    Arc::new(tokio::sync::Mutex::new(HashMap::<String, String>::new()));

                // Register memory tools
                let memory_store_clone = Arc::clone(&memory_store);
                let set_tool = Tool::new(
                    "memory_set".to_string(),
                    "Set a value in memory".to_string(),
                    json!({
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "Key to store the value under"
                            },
                            "value": {
                                "type": "string",
                                "description": "Value to store"
                            }
                        },
                        "required": ["key", "value"]
                    }),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "success": {
                                "type": "boolean",
                                "description": "Whether the operation was successful"
                            }
                        }
                    })),
                    Box::new(move |args| {
                        let memory_store_clone = memory_store_clone.clone();
                        Box::pin(async move {
                            info!("memory_set called with args: {:?}", args);
                            let key = args["key"]
                                .as_str()
                                .ok_or_else(|| Error::msg("Invalid argument: key"))?;
                            let value = args["value"]
                                .as_str()
                                .ok_or_else(|| Error::msg("Invalid argument: value"))?;
                            info!("memory_set storing key: {}, value: {}", key, value);

                            let mut store = memory_store_clone.lock().await;
                            store.insert(key.to_string(), value.to_string());
                            info!("memory_set store contents after insert: {:?}", *store);
                            Ok(json!({ "success": true }))
                        })
                    }),
                );
                server.mcp_server.register_tool(set_tool).await?;

                let memory_store_clone = Arc::clone(&memory_store);
                let get_tool = Tool::new(
                    "memory_get".to_string(),
                    "Get a value from memory".to_string(),
                    json!({
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "Key to retrieve the value for"
                            }
                        },
                        "required": ["key"]
                    }),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "string",
                                "description": "The retrieved value"
                            },
                            "exists": {
                                "type": "boolean",
                                "description": "Whether the key exists"
                            }
                        }
                    })),
                    Box::new(move |args| {
                        let memory_store_clone = memory_store_clone.clone();
                        Box::pin(async move {
                            info!("memory_get called with args: {:?}", args);
                            let key = args["key"]
                                .as_str()
                                .ok_or_else(|| Error::msg("Invalid argument: key"))?;
                            info!("memory_get looking for key: {}", key);

                            let store = memory_store_clone.lock().await;
                            info!("memory_get store contents: {:?}", *store);
                            if let Some(value) = store.get(key) {
                                info!("memory_get found value: {}", value);
                                Ok(json!({
                                    "value": value.clone(),
                                    "exists": true
                                }))
                            } else {
                                info!("memory_get key not found: {}", key);
                                Ok(json!({
                                    "value": "",
                                    "exists": false
                                }))
                            }
                        })
                    }),
                );
                server.mcp_server.register_tool(get_tool).await?;

                // Register memory resources
                let memory_store_clone = memory_store.clone();
                let memory_resource = Resource::new(
                    "memory_store".to_string(),
                    "In-memory key-value store".to_string(),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "Action to perform (get/set/list)",
                                "enum": ["get", "set", "list"]
                            },
                            "key": {
                                "type": "string",
                                "description": "Key to access"
                            },
                            "value": {
                                "type": "string",
                                "description": "Value to set"
                            }
                        },
                        "required": ["action"]
                    })),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "size": {
                                "type": "integer",
                                "description": "Number of items in the store"
                            },
                            "keys": {
                                "type": "array",
                                "description": "List of all keys in the store",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "value": {
                                "type": "string",
                                "description": "Retrieved value"
                            },
                            "success": {
                                "type": "boolean",
                                "description": "Whether the operation was successful"
                            }
                        }
                    })),
                    Box::new(move |args| {
                        let store = memory_store_clone.clone();
                        Box::pin(async move {
                            match args.get("action").and_then(|v| v.as_str()) {
                                Some("list") => {
                                    let store = store.lock().await;
                                    Ok(json!({
                                        "size": store.len(),
                                        "keys": store.keys().cloned().collect::<Vec<_>>(),
                                        "success": true
                                    }))
                                }
                                Some("get") => {
                                    let key = args
                                        .get("key")
                                        .and_then(|v| v.as_str())
                                        .ok_or_else(|| Error::msg("Missing key argument"))?;
                                    let store = store.lock().await;
                                    Ok(json!({
                                        "value": store.get(key).cloned().unwrap_or_default(),
                                        "success": store.contains_key(key)
                                    }))
                                }
                                Some("set") => {
                                    let key = args
                                        .get("key")
                                        .and_then(|v| v.as_str())
                                        .ok_or_else(|| Error::msg("Missing key argument"))?;
                                    let value = args
                                        .get("value")
                                        .and_then(|v| v.as_str())
                                        .ok_or_else(|| Error::msg("Missing value argument"))?;
                                    let mut store = store.lock().await;
                                    store.insert(key.to_string(), value.to_string());
                                    Ok(json!({ "success": true }))
                                }
                                _ => Err(Error::msg("Invalid action")),
                            }
                        })
                    }),
                );
                server.mcp_server.register_resource(memory_resource).await?;

                // Store the memory store in the server
                server.memory_store = Some(memory_store);
            }
            "sqlite" => {
                info!("Registering SQLite tools");
                // Register SQLite tools
                let execute_tool = Tool::new(
                    "sqlite_execute".to_string(),
                    "Execute a SQL query".to_string(),
                    json!({
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "SQL query to execute"
                            }
                        },
                        "required": ["query"]
                    }),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "success": {
                                "type": "boolean",
                                "description": "Whether the query executed successfully"
                            },
                            "rows_affected": {
                                "type": "integer",
                                "description": "Number of rows affected by the query"
                            }
                        }
                    })),
                    Box::new(|_| {
                        Box::pin(async move { Ok(json!({ "success": true, "rows_affected": 0 })) })
                    }),
                );
                server.mcp_server.register_tool(execute_tool).await?;
                info!("Registered sqlite_execute tool");

                let query_tool = Tool::new(
                    "sqlite_query".to_string(),
                    "Execute a SQL query and return results".to_string(),
                    json!({
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "SQL query to execute"
                            }
                        },
                        "required": ["query"]
                    }),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "success": {
                                "type": "boolean",
                                "description": "Whether the query executed successfully"
                            },
                            "rows": {
                                "type": "array",
                                "description": "Query results as an array of objects",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": true
                                }
                            }
                        }
                    })),
                    Box::new(|_| {
                        Box::pin(async move { Ok(json!({ "success": true, "rows": [] })) })
                    }),
                );
                server.mcp_server.register_tool(query_tool).await?;
                info!("Registered sqlite_query tool");

                let create_table_tool = Tool::new(
                    "sqlite_create_table".to_string(),
                    "Create a new table".to_string(),
                    json!({
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name of the table to create"
                            },
                            "columns": {
                                "type": "array",
                                "description": "Array of column definitions",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Column name"
                                        },
                                        "type": {
                                            "type": "string",
                                            "description": "Column type (INTEGER, TEXT, REAL, BLOB)"
                                        },
                                        "primary_key": {
                                            "type": "boolean",
                                            "description": "Whether this column is a primary key"
                                        },
                                        "not_null": {
                                            "type": "boolean",
                                            "description": "Whether this column cannot be null"
                                        },
                                        "unique": {
                                            "type": "boolean",
                                            "description": "Whether this column must be unique"
                                        }
                                    },
                                    "required": ["name", "type"]
                                }
                            }
                        },
                        "required": ["name", "columns"]
                    }),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "success": {
                                "type": "boolean",
                                "description": "Whether the table was created successfully"
                            }
                        }
                    })),
                    Box::new(|_| Box::pin(async move { Ok(json!({ "success": true })) })),
                );
                server.mcp_server.register_tool(create_table_tool).await?;
                info!("Registered sqlite_create_table tool");

                let drop_table_tool = Tool::new(
                    "sqlite_drop_table".to_string(),
                    "Drop a table".to_string(),
                    json!({
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name of the table to drop"
                            }
                        },
                        "required": ["name"]
                    }),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "success": {
                                "type": "boolean",
                                "description": "Whether the table was dropped successfully"
                            }
                        }
                    })),
                    Box::new(|_| Box::pin(async move { Ok(json!({ "success": true })) })),
                );
                server.mcp_server.register_tool(drop_table_tool).await?;
                info!("Registered sqlite_drop_table tool");

                let list_tables_tool = Tool::new(
                    "sqlite_list_tables".to_string(),
                    "List all tables in the database".to_string(),
                    json!({
                        "type": "object",
                        "properties": {}
                    }),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "tables": {
                                "type": "array",
                                "description": "List of table names",
                                "items": {
                                    "type": "string"
                                }
                            }
                        }
                    })),
                    Box::new(|_| Box::pin(async move { Ok(json!({ "tables": [] })) })),
                );
                server.mcp_server.register_tool(list_tables_tool).await?;
                info!("Registered sqlite_list_tables tool");

                // Register SQLite resources
                let sqlite_resource = Resource::new(
                    "sqlite_database".to_string(),
                    "SQLite database connection".to_string(),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "Action to perform (tables/size)",
                                "enum": ["tables", "size"]
                            }
                        },
                        "required": ["action"]
                    })),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "tables": {
                                "type": "array",
                                "description": "List of tables in the database",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "size": {
                                "type": "integer",
                                "description": "Size of the database file in bytes"
                            }
                        }
                    })),
                    Box::new(|args| {
                        Box::pin(async move {
                            match args.get("action").and_then(|v| v.as_str()) {
                                Some("tables") => Ok(json!({
                                    "tables": Vec::<String>::new()
                                })),
                                Some("size") => Ok(json!({
                                    "size": 0
                                })),
                                _ => Err(Error::msg("Invalid action")),
                            }
                        })
                    }),
                );
                server.mcp_server.register_resource(sqlite_resource).await?;

                // Register file system resources
                let filesystem_resource = Resource::new(
                    "filesystem".to_string(),
                    "Local file system access".to_string(),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "Action to perform (cwd/space)",
                                "enum": ["cwd", "space"]
                            }
                        },
                        "required": ["action"]
                    })),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "current_directory": {
                                "type": "string",
                                "description": "Current working directory"
                            },
                            "available_space": {
                                "type": "integer",
                                "description": "Available disk space in bytes"
                            }
                        }
                    })),
                    Box::new(|args| {
                        Box::pin(async move {
                            match args.get("action").and_then(|v| v.as_str()) {
                                Some("cwd") => {
                                    let cwd = std::env::current_dir()
                                        .unwrap_or_default()
                                        .to_string_lossy()
                                        .to_string();
                                    Ok(json!({
                                        "current_directory": cwd
                                    }))
                                }
                                Some("space") => Ok(json!({
                                    "available_space": 0 // TODO: Implement actual disk space check
                                })),
                                _ => Err(Error::msg("Invalid action")),
                            }
                        })
                    }),
                );
                server
                    .mcp_server
                    .register_resource(filesystem_resource)
                    .await?;

                // Register browser resources
                let browser_resource = Resource::new(
                    "browser".to_string(),
                    "Browser automation resources".to_string(),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "Action to perform (url/viewport)",
                                "enum": ["url", "viewport"]
                            }
                        },
                        "required": ["action"]
                    })),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "current_url": {
                                "type": "string",
                                "description": "Current page URL"
                            },
                            "viewport": {
                                "type": "object",
                                "description": "Browser viewport dimensions",
                                "properties": {
                                    "width": {
                                        "type": "integer",
                                        "description": "Viewport width in pixels"
                                    },
                                    "height": {
                                        "type": "integer",
                                        "description": "Viewport height in pixels"
                                    }
                                }
                            }
                        }
                    })),
                    Box::new(|args| {
                        Box::pin(async move {
                            match args.get("action").and_then(|v| v.as_str()) {
                                Some("url") => Ok(json!({
                                    "current_url": "about:blank"
                                })),
                                Some("viewport") => Ok(json!({
                                    "viewport": {
                                        "width": 1024,
                                        "height": 768
                                    }
                                })),
                                _ => Err(Error::msg("Invalid action")),
                            }
                        })
                    }),
                );
                server
                    .mcp_server
                    .register_resource(browser_resource)
                    .await?;
            }
            "file" => {
                // Register file tools
                let read_tool = Tool::new(
                    "file_read".to_string(),
                    "Read contents of a file".to_string(),
                    json!({
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            }
                        },
                        "required": ["path"]
                    }),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Contents of the file"
                            },
                            "exists": {
                                "type": "boolean",
                                "description": "Whether the file exists"
                            }
                        }
                    })),
                    Box::new(|_| {
                        Box::pin(async move { Ok(json!({ "content": "", "exists": false })) })
                    }),
                );
                server.mcp_server.register_tool(read_tool).await?;

                let write_tool = Tool::new(
                    "file_write".to_string(),
                    "Write contents to a file".to_string(),
                    json!({
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["path", "content"]
                    }),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "success": {
                                "type": "boolean",
                                "description": "Whether the write was successful"
                            }
                        }
                    })),
                    Box::new(|_| Box::pin(async move { Ok(json!({ "success": true })) })),
                );
                server.mcp_server.register_tool(write_tool).await?;

                // Register list directory tool
                let list_tool = Tool::new(
                    "list_directory".to_string(),
                    "List contents of a directory".to_string(),
                    json!({
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the directory to list (relative to root)"
                            }
                        },
                        "required": ["path"]
                    }),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "entries": {
                                "type": "array",
                                "description": "List of directory entries",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Name of the entry"
                                        },
                                        "type": {
                                            "type": "string",
                                            "description": "Type of the entry (file/directory)"
                                        },
                                        "size": {
                                            "type": "integer",
                                            "description": "Size of the entry in bytes (for files)"
                                        },
                                        "modified": {
                                            "type": "string",
                                            "description": "Last modification time"
                                        }
                                    }
                                }
                            }
                        }
                    })),
                    Box::new(|_| Box::pin(async move { Ok(json!({ "entries": [] })) })),
                );
                server.mcp_server.register_tool(list_tool).await?;

                // Register file system resources
                let filesystem_resource = Resource::new(
                    "filesystem".to_string(),
                    "Local file system access".to_string(),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "Action to perform (cwd/space)",
                                "enum": ["cwd", "space"]
                            }
                        },
                        "required": ["action"]
                    })),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "current_directory": {
                                "type": "string",
                                "description": "Current working directory"
                            },
                            "available_space": {
                                "type": "integer",
                                "description": "Available disk space in bytes"
                            }
                        }
                    })),
                    Box::new(|args| {
                        Box::pin(async move {
                            match args.get("action").and_then(|v| v.as_str()) {
                                Some("cwd") => Ok(json!({
                                    "current_directory": std::env::current_dir()
                                        .unwrap_or_default()
                                        .to_string_lossy()
                                        .to_string()
                                })),
                                Some("space") => Ok(json!({
                                    "available_space": 0 // TODO: Implement actual disk space check
                                })),
                                _ => Err(Error::msg("Invalid action")),
                            }
                        })
                    }),
                );
                server
                    .mcp_server
                    .register_resource(filesystem_resource)
                    .await?;
            }
            "puppeteer" => {
                info!("Registering puppeteer tools");
                // Register puppeteer tools
                let navigate_tool = Tool::new(
                    "puppeteer_navigate".to_string(),
                    "Navigate to a URL".to_string(),
                    json!({
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL to navigate to"
                            }
                        },
                        "required": ["url"]
                    }),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "success": {
                                "type": "boolean",
                                "description": "Whether navigation was successful"
                            }
                        }
                    })),
                    Box::new(|_| Box::pin(async move { Ok(json!({ "success": true })) })),
                );
                server.mcp_server.register_tool(navigate_tool).await?;
                info!("Registered puppeteer_navigate tool");

                let click_tool = Tool::new(
                    "puppeteer_click".to_string(),
                    "Click an element".to_string(),
                    json!({
                        "type": "object",
                        "properties": {
                            "selector": {
                                "type": "string",
                                "description": "CSS selector for the element to click"
                            }
                        },
                        "required": ["selector"]
                    }),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "success": {
                                "type": "boolean",
                                "description": "Whether click was successful"
                            }
                        }
                    })),
                    Box::new(|_| Box::pin(async move { Ok(json!({ "success": true })) })),
                );
                server.mcp_server.register_tool(click_tool).await?;
                info!("Registered puppeteer_click tool");

                let type_tool = Tool::new(
                    "puppeteer_type".to_string(),
                    "Type text into an element".to_string(),
                    json!({
                        "type": "object",
                        "properties": {
                            "selector": {
                                "type": "string",
                                "description": "CSS selector for the element to type into"
                            },
                            "text": {
                                "type": "string",
                                "description": "Text to type"
                            }
                        },
                        "required": ["selector", "text"]
                    }),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "success": {
                                "type": "boolean",
                                "description": "Whether typing was successful"
                            }
                        }
                    })),
                    Box::new(|_| Box::pin(async move { Ok(json!({ "success": true })) })),
                );
                server.mcp_server.register_tool(type_tool).await?;
                info!("Registered puppeteer_type tool");

                let screenshot_tool = Tool::new(
                    "puppeteer_screenshot".to_string(),
                    "Take a screenshot".to_string(),
                    json!({
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to save the screenshot"
                            }
                        },
                        "required": ["path"]
                    }),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "success": {
                                "type": "boolean",
                                "description": "Whether screenshot was successful"
                            }
                        }
                    })),
                    Box::new(|_| Box::pin(async move { Ok(json!({ "success": true })) })),
                );
                server.mcp_server.register_tool(screenshot_tool).await?;
                info!("Registered puppeteer_screenshot tool");

                // Register browser resources
                let browser_resource = Resource::new(
                    "browser".to_string(),
                    "Browser automation resources".to_string(),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "Action to perform (url/viewport)",
                                "enum": ["url", "viewport"]
                            }
                        },
                        "required": ["action"]
                    })),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "current_url": {
                                "type": "string",
                                "description": "Current page URL"
                            },
                            "viewport": {
                                "type": "object",
                                "description": "Browser viewport dimensions",
                                "properties": {
                                    "width": {
                                        "type": "integer",
                                        "description": "Viewport width in pixels"
                                    },
                                    "height": {
                                        "type": "integer",
                                        "description": "Viewport height in pixels"
                                    }
                                }
                            }
                        }
                    })),
                    Box::new(|args| {
                        Box::pin(async move {
                            match args.get("action").and_then(|v| v.as_str()) {
                                Some("url") => Ok(json!({
                                    "current_url": "about:blank"
                                })),
                                Some("viewport") => Ok(json!({
                                    "viewport": {
                                        "width": 1024,
                                        "height": 768
                                    }
                                })),
                                _ => Err(Error::msg("Invalid action")),
                            }
                        })
                    }),
                );
                server
                    .mcp_server
                    .register_resource(browser_resource)
                    .await?;
            }
            _ => {}
        }

        info!("Server {} initialized successfully", name);
        Ok(server)
    }

    async fn list_tools(&self) -> Result<Vec<ToolSchema>> {
        debug!("Listing tools for server: {}", self.name);
        let tools = self.mcp_server.list_tools().await?;
        info!("Found {} tools for server {}", tools.len(), self.name);
        Ok(tools)
    }

    async fn list_resources(&self) -> Result<Vec<crate::protocol::ResourceSchema>> {
        debug!("Listing resources for server: {}", self.name);
        let resources = self.mcp_server.list_resources().await?;
        info!(
            "Found {} resources for server {}",
            resources.len(),
            self.name
        );
        Ok(resources)
    }

    async fn execute_tool(
        &mut self,
        tool_name: &str,
        arguments: &Value,
        retries: u32,
        delay: Duration,
    ) -> Result<Value> {
        info!("Executing tool {} on server {}", tool_name, self.name);
        debug!("Tool arguments: {}", arguments);
        let mut attempt = 0;

        while attempt < retries {
            match self
                .mcp_server
                .execute_tool(tool_name, arguments, 3, Duration::from_secs(1))
                .await
            {
                Ok(result) => {
                    info!("Tool {} executed successfully", tool_name);
                    debug!("Tool result: {}", result);
                    return Ok(result);
                }
                Err(e) => {
                    attempt += 1;
                    if attempt < retries {
                        warn!(
                            "Error executing tool. Attempt {} of {}: {}",
                            attempt, retries, e
                        );
                        sleep(delay).await;
                    } else {
                        error!("Max retries reached for tool {}", tool_name);
                        return Err(e);
                    }
                }
            }
        }

        Err(Error::msg("Max retries reached"))
    }

    async fn cleanup(&mut self) -> Result<()> {
        info!("Cleaning up server: {}", self.name);
        let _lock = self.cleanup_lock.lock().await;
        self.mcp_server.cleanup().await?;
        info!("Server {} cleaned up successfully", self.name);
        Ok(())
    }
}

/// Structure to hold all system prompts
#[derive(Debug, serde::Deserialize)]
struct SystemPrompts {
    server_prompts: std::collections::HashMap<String, ServerPrompt>,
    default_system_prompt: String,
    common_prompts: CommonPrompts,
    commands: std::collections::HashMap<String, String>,
    tool_examples: std::collections::HashMap<String, Vec<ToolExample>>,
}

#[derive(Debug, serde::Deserialize)]
struct ServerPrompt {
    system_prompt: String,
}

#[derive(Debug, serde::Deserialize)]
struct CommonPrompts {
    tool_response: String,
    welcome: String,
    error_prompts: ErrorPrompts,
}

#[derive(Debug, serde::Deserialize)]
struct ErrorPrompts {
    ollama_not_found: String,
    ollama_connection_failed: String,
}

#[derive(Debug, serde::Deserialize)]
struct ToolExample {
    description: String,
    example: String,
}

/// Loads system prompts from YAML file or returns default prompts
fn load_system_prompts() -> SystemPrompts {
    match fs::read_to_string("mcp_prompts.yaml") {
        Ok(content) => match serde_yaml::from_str::<SystemPrompts>(&content) {
            Ok(prompts) => {
                info!("Successfully loaded prompts from mcp_prompts.yaml");
                prompts
            }
            Err(e) => {
                warn!(
                    "Failed to parse mcp_prompts.yaml: {}. Using default prompts.",
                    e
                );
                create_default_prompts()
            }
        },
        Err(e) => {
            warn!(
                "Failed to read mcp_prompts.yaml: {}. Using default prompts.",
                e
            );
            create_default_prompts()
        }
    }
}

/// Creates default system prompts
fn create_default_prompts() -> SystemPrompts {
    let mut commands = std::collections::HashMap::new();
    commands.insert(
        "help".to_string(),
        "Display help menu and available commands".to_string(),
    );
    commands.insert("clear".to_string(), "Clear the terminal screen".to_string());
    commands.insert("usage".to_string(), "Display usage information".to_string());
    commands.insert("exit".to_string(), "Exit the program".to_string());
    commands.insert(
        "servers".to_string(),
        "List available MCP servers".to_string(),
    );
    commands.insert("tools".to_string(), "List available tools".to_string());
    commands.insert(
        "resources".to_string(),
        "List available resources".to_string(),
    );
    commands.insert(
        "debug".to_string(),
        "Toggle debug logging level".to_string(),
    );
    commands.insert(
        "ai".to_string(),
        "Switch between AI providers (Ollama/OpenAI)".to_string(),
    );

    let mut tool_examples = std::collections::HashMap::new();
    tool_examples.insert(
        "memory".to_string(),
        vec![
            ToolExample {
                description: "Store a value in memory".to_string(),
                example: r#"{"tool": "memory_set", "arguments": {"key": "name", "value": "John"}}"#
                    .to_string(),
            },
            ToolExample {
                description: "Retrieve a value from memory".to_string(),
                example: r#"{"tool": "memory_get", "arguments": {"key": "name"}}"#.to_string(),
            },
        ],
    );

    SystemPrompts {
        server_prompts: std::collections::HashMap::new(),
        default_system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
        common_prompts: CommonPrompts {
            tool_response: DEFAULT_TOOL_RESPONSE_PROMPT.to_string(),
            welcome: DEFAULT_WELCOME_MESSAGE.to_string(),
            error_prompts: ErrorPrompts {
                ollama_not_found: "Ollama model not found, please check:\n1. Make sure llama3.2:latest model is downloaded: ollama pull llama3.2:latest\n2. Switch to default Deepseek API? (y/n):".to_string(),
                ollama_connection_failed: "Ollama server connection failed, please check:\n1. Make sure Ollama service is running\n2. Check if Ollama service is running at http://localhost:11434\n3. Make sure llama3.2:latest model is downloaded: ollama pull llama3.2:latest\n4. Switch to default Deepseek API? (y/n):".to_string(),
            },
        },
        commands,
        tool_examples,
    }
}

/// Represents a chat session with the LLM
/// Manages the conversation state, tools, and server interactions
struct ChatSession {
    llm_client: Arc<Mutex<LLMClient>>, // Thread-safe LLM client
    servers: Vec<Server>,              // List of available servers
    history: History,                  // Chat history
    current_ai_server: String,         // Currently active AI server
    running: bool,                     // Session running state
    readline: Editor<(), FileHistory>, // Command line editor
    cached_tools: Vec<ToolSchema>,     // Cached tool definitions
    prompts: SystemPrompts,            // Add prompts field
    rag_server: Arc<Mutex<RagServer>>, // Thread-safe RagServer instance
}

impl ChatSession {
    /// Creates a new chat session
    /// Initializes the LLM client, servers, and command line interface
    pub async fn new(llm_client: Arc<Mutex<LLMClient>>) -> Result<Self> {
        let mut servers = Vec::new();
        let config = Configuration::new();
        let server_config = config.load_config("config/servers_config.json")?;

        if let Some(servers_config) = server_config.get("mcpServers") {
            for (name, config) in servers_config.as_object().unwrap() {
                let server = Server::new(name.clone(), config.clone()).await?;
                servers.push(server);
            }
        }

        // Initialize command line editor with history
        let mut readline = Editor::<(), FileHistory>::new()?;
        readline.set_max_history_size(1000)?;
        if let Err(e) = readline.load_history(".mcp_history") {
            info!("No history file found: {}", e);
        }

        // Cache tool definitions for quick access
        let mut cached_tools = Vec::new();
        for server in &servers {
            let tools = server.list_tools().await?;
            cached_tools.extend(tools);
        }

        // Initialize RagServer
        let rag_server = Arc::new(Mutex::new(RagServer::new()?));
        {
            let mut rag = rag_server.lock().await;
            rag.init().await?;
        }

        Ok(Self {
            llm_client,
            servers,
            history: History::new(),
            current_ai_server: "ollama".to_string(),
            running: true,
            readline,
            cached_tools,
            prompts: load_system_prompts(), // Load prompts
            rag_server,
        })
    }

    /// Cleans up all servers when the session ends
    async fn cleanup_servers(&mut self) -> Result<()> {
        info!("Cleaning up all servers");
        for server in &mut self.servers {
            if let Err(e) = server.cleanup().await {
                warn!("Warning during server cleanup: {}", e);
            }
        }
        info!("All servers cleaned up");
        Ok(())
    }

    /// Gets the appropriate system prompt for the current server
    fn get_system_prompt(&self, server_name: &str) -> String {
        if let Some(server_prompt) = self.prompts.server_prompts.get(server_name) {
            server_prompt.system_prompt.clone()
        } else {
            self.prompts.default_system_prompt.clone()
        }
    }

    /// Processes an LLM response
    /// Handles both tool calls and regular responses
    async fn process_llm_response(&mut self, input: &str) -> Result<String> {
        let stop_signal = Arc::new(Mutex::new(false));
        let stop_signal_clone = Arc::clone(&stop_signal);
        let thinking_handle = tokio::spawn(async move {
            print_bot_thinking_continuous(stop_signal_clone).await;
        });

        // Format input as a message array with system message
        let mut messages = Vec::new();

        // Add system message with tool descriptions
        let mut tools_description = String::new();
        for tool in &self.cached_tools {
            tools_description.push_str(&format!("{}\n", tool.format_for_llm()));
        }

        // Get the appropriate system prompt based on the server
        let system_prompt = self.get_system_prompt(&self.current_ai_server);

        let system_message = Message {
            role: "system".to_string(),
            content: system_prompt,
        };

        messages.push(json!({
            "role": "system",
            "content": system_message
        }));

        // Add user message
        messages.push(json!({
            "role": "user",
            "content": input
        }));

        debug!("Processing user input: {}", input);
        debug!(
            "Formatted messages: {}",
            serde_json::to_string_pretty(&messages)?
        );

        // Get the first LLM response
        let mut llm_client = self.llm_client.lock().await;
        let response = llm_client
            .get_response(&messages, Some("deepseek-chat"))
            .await?;
        drop(llm_client);

        // Try to parse the response as a tool call
        match serde_json::from_str::<Value>(&response) {
            Ok(tool_call)
                if tool_call.get("tool").is_some() && tool_call.get("arguments").is_some() =>
            {
                // It's a tool call, process it
                let tool_name = tool_call["tool"]
                    .as_str()
                    .ok_or_else(|| Error::msg("Invalid tool name"))?;
                let arguments = &tool_call["arguments"];

                debug!("Executing tool: {}", tool_name);
                debug!(
                    "Tool arguments: {}",
                    serde_json::to_string_pretty(arguments)?
                );

                // Let LLM process the tool response
                let tool_response = serde_json::to_string(&arguments)?;
                let response_messages = vec![
                    json!({
                        "role": "system",
                        "content": self.prompts.common_prompts.tool_response.clone()
                    }),
                    json!({
                        "role": "user",
                        "content": format!("Please explain this tool response: {}", tool_response)
                    }),
                ];

                // Get the second LLM response
                let mut llm_client = self.llm_client.lock().await;
                let processed_response = llm_client
                    .get_response(&response_messages, Some("deepseek-chat"))
                    .await?;

                // Stop the thinking animation
                *stop_signal.lock().await = true;
                thinking_handle.await?;

                Ok(processed_response)
            }
            _ => {
                // It's a normal response, return it directly
                *stop_signal.lock().await = true;
                thinking_handle.await?;
                Ok(response)
            }
        }
    }

    /// Processes a command entered by the user
    /// Handles various built-in commands like /help, /clear, etc.
    async fn process_command(&mut self, command: &str) -> Result<()> {
        match command {
            "/help" => {
                self.show_help();
            }
            "/clear" => {
                print!("\x1B[2J\x1B[H");
                io::stdout().flush()?;
            }
            "/usage" => {
                print_about();
            }
            "/exit" => {
                self.running = false;
            }
            "/servers" | "/mcp-servers" => {
                let config = Configuration::new();
                let server_config = config.load_config("config/servers_config.json")?;
                print_mcp_servers(&server_config);
            }
            "/tools" => {
                for server in &self.servers {
                    let tools = server.list_tools().await?;
                    print_colored_ln(
                        &format!("\nTools from {} server:", server.name),
                        Color::Cyan,
                    );
                    print_tools(&tools);
                }
            }
            "/resources" => {
                for server in &self.servers {
                    let resources = server.list_resources().await?;
                    print_colored_ln(
                        &format!("\nResources from {} server:", server.name),
                        Color::Cyan,
                    );
                    print_resources(&resources);
                }
            }
            "/debug" => {
                // Toggle debug logging level
                let current_level = log::max_level();
                let new_level = if current_level == log::LevelFilter::Info {
                    log::LevelFilter::Debug
                } else {
                    log::LevelFilter::Info
                };
                log::set_max_level(new_level);
                print_colored_ln(
                    &format!(
                        "Debug information is now {}",
                        if new_level == log::LevelFilter::Debug {
                            "enabled"
                        } else {
                            "disabled"
                        }
                    ),
                    Color::Green,
                );
            }
            "/ai" => {
                if self.current_ai_server == "ollama" {
                    // Switch to OpenAI when API key is available
                    if let Ok(api_key) = env::var("OPENAI_API_KEY") {
                        self.current_ai_server = "openai".to_string();
                        let config = Configuration::new();
                        let mut llm_config = config.load_llm_config("config/config.yaml")?;
                        llm_config.api_keys.openai = api_key;
                        let mut llm_client = self.llm_client.lock().await;
                        *llm_client = LLMClient::new(llm_config, LLMProvider::OpenAI);
                        print_colored_ln("Switched to OpenAI server", Color::Green);
                    } else {
                        print_colored_ln(
                            "Error: OPENAI_API_KEY not found in environment variables",
                            Color::Red,
                        );
                        print_colored_ln(
                            "\nTo use OpenAI, please set the OPENAI_API_KEY environment variable:",
                            Color::Yellow,
                        );
                        print_colored_ln("export OPENAI_API_KEY='your-api-key-here'", Color::Green);
                    }
                } else {
                    // Switch back to Ollama
                    self.current_ai_server = "ollama".to_string();
                    let config = Configuration::new();
                    let llm_config = config.load_llm_config("config/config.yaml")?;
                    let mut llm_client = self.llm_client.lock().await;
                    *llm_client = LLMClient::new(llm_config, LLMProvider::Deepseek);
                    print_colored_ln("Switched to Ollama server", Color::Green);
                }
            }
            "/voice" => {
                print_colored_ln(
                    "Starting voice recording... (Press Enter to stop)",
                    Color::Yellow,
                );
                let whisper = WhisperServer::new()?;
                whisper.start_recording().await?;

                // Wait for Enter key
                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;

                let text = whisper.stop_recording().await?;
                print_colored_ln(&format!("Recognized text: {}", text), Color::Green);

                // Process the recognized text
                self.process_message(&text).await?;
            }
            "/rag-add" => {
                print_colored_ln(
                    "Enter document text (press Ctrl+D when done):",
                    Color::Yellow,
                );
                let mut text = String::new();
                std::io::stdin().read_to_string(&mut text)?;

                let doc = Document {
                    id: uuid::Uuid::new_v4().to_string(),
                    text,
                    metadata: Some(serde_json::json!({
                        "added_at": chrono::Utc::now().to_rfc3339(),
                        "source": "manual_input"
                    })),
                };
                let mut rag = self.rag_server.lock().await;
                rag.add_document(doc).await?;
                print_colored_ln("Document added successfully!", Color::Green);
            }
            "/rag-search" => {
                print_colored_ln("Enter search query:", Color::Yellow);
                let mut query = String::new();
                std::io::stdin().read_line(&mut query)?;
                query = query.trim().to_string();

                print_colored_ln("Enter number of results (default: 5):", Color::Yellow);
                let mut limit_str = String::new();
                std::io::stdin().read_line(&mut limit_str)?;
                let limit = limit_str.trim().parse::<u32>().unwrap_or(5);

                let rag = self.rag_server.lock().await;
                let results = rag.search_similar(&query, limit, None).await?;

                print_colored_ln("\nFound similar documents:", Color::Green);
                for (i, result) in results.iter().enumerate() {
                    println!("\n--- Result {} (Score: {:.3}) ---", i + 1, result.score);
                    println!("Document ID: {}", result.document_id);
                    println!("Chunk Index: {}", result.chunk_index);
                    if let Some(metadata) = &result.metadata {
                        println!("Metadata: {}", serde_json::to_string_pretty(metadata)?);
                    }
                    println!("Text: {}", result.text);
                }
            }
            "/rag-info" => {
                let info = self.rag_server.lock().await.get_collection_info().await?;
                print_colored_ln("\nRAG Database Information:", Color::Green);
                println!("Collection Name: {}", info["name"]);
                println!("Vector Size: {}", info["vector_size"]);
                println!("Distance Metric: {}", info["distance"]);
                println!("Total Documents: {}", info["points_count"]);
            }
            _ => {
                println!("Unknown command: {}", command);
            }
        }
        Ok(())
    }

    /// Processes a user message
    /// Handles both regular messages and tool calls
    async fn process_message(&mut self, input: &str) -> Result<()> {
        let stop_signal = Arc::new(Mutex::new(false));
        let stop_signal_clone = Arc::clone(&stop_signal);
        let thinking_handle = tokio::spawn(async move {
            print_bot_thinking_continuous(stop_signal_clone).await;
        });

        // Check for CTRL+K (stop inference)
        if input.trim() == "\x0B" {
            print_colored_ln("\nStopping inference...", Color::Yellow);
            *stop_signal.lock().await = true;
            thinking_handle.await?;
            return Ok(());
        }

        match self.process_llm_response(input).await {
            Ok(response) => {
                // Stop thinking animation and display response
                *stop_signal.lock().await = true;
                thinking_handle.await?;

                print!("\r\x1b[K\x1b[33m@AI-BOT: \x1b[0m");
                io::stdout().flush()?;
                typing_animation(&response, 20);
            }
            Err(e) => {
                // Stop thinking animation and display error
                *stop_signal.lock().await = true;
                thinking_handle.await?;

                print!("\r\x1b[K");
                io::stdout().flush()?;
                print_colored_ln(&format!("Error: {}", e), Color::Red);
            }
        }
        Ok(())
    }

    /// Starts the chat session
    /// Main loop for handling user input and commands
    async fn start(&mut self) -> Result<()> {
        // Display menu (only once)
        print_menu(&self.current_ai_server).await;

        // Display welcome message from prompts
        print_colored_ln(&self.prompts.common_prompts.welcome, Color::Yellow);
        let model_name = if self.current_ai_server == "ollama" {
            "llama3.2:latest"
        } else {
            "gpt-3.5-turbo"
        };
        print_colored_ln(
            &format!(
                "Current AI Server: {} ({})",
                self.current_ai_server, model_name
            ),
            Color::Cyan,
        );
        println!();

        while self.running {
            match self.readline.readline("\n\x1b[34m@Human: \x1b[0m") {
                Ok(line) => {
                    if line.trim().is_empty() {
                        continue;
                    }
                    // Save to history
                    self.readline.add_history_entry(line.trim());
                    if let Err(e) = self.readline.save_history(".mcp_history") {
                        error!("Failed to save history: {}", e);
                    }

                    if line.starts_with('/') {
                        self.process_command(line.trim()).await?;
                    } else {
                        self.process_message(line.trim()).await?;
                    }
                }
                Err(rustyline::error::ReadlineError::Interrupted) => {
                    // CTRL+C pressed
                    print_colored_ln("\nExiting program...", Color::Yellow);
                    self.running = false;
                }
                Err(rustyline::error::ReadlineError::Eof) => {
                    // CTRL+D pressed
                    print_colored_ln("\nExiting program...", Color::Yellow);
                    self.running = false;
                }
                Err(e) => {
                    error!("Error reading line: {}", e);
                    break;
                }
            }
        }
        Ok(())
    }

    /// Initializes all servers from configuration
    async fn initialize_servers(&self) -> Result<Value> {
        let config = Configuration::new();
        let server_config = config.load_config("config/servers_config.json")?;

        // Validate configuration format
        if let Some(_servers) = server_config.get("mcpServers") {
            Ok(server_config)
        } else {
            Err(Error::msg(
                "Invalid server configuration: missing mcpServers section",
            ))
        }
    }

    fn show_help(&self) {
        println!("\nAvailable commands:");
        println!("  /help     - Show this help message");
        println!("  /clear    - Clear the chat history");
        println!("  /usage    - Show token usage statistics");
        println!("  /exit     - Exit the program");
        println!("  /servers  - Show available LLM servers");
        println!("  /tools    - Show available tools");
        println!("  /resources - Show available resources");
        println!("  /voice    - Start voice input (press Enter to stop recording)");
        println!("  /rag-add   - Add a new document to RAG database");
        println!("  /rag-search - Search for similar documents");
        println!("  /rag-info  - Show RAG database information");
        println!("\nYou can also use these commands in your messages:");
        println!("  /debug    - Toggle debug mode");
        println!("  /model    - Switch LLM model");
        println!("  /system   - Set system prompt");
        println!("\nType your message and press Enter to send.");
    }
}

/// Reads a single character from stdin
fn read_char() -> Result<char> {
    let mut input = [0u8; 1];
    io::stdin().read_exact(&mut input)?;
    Ok(input[0] as char)
}

/// Prepares messages for tool calls
/// Formats the tool call and response for LLM processing
fn prepare_tool_call_messages(response: &str) -> Result<Vec<Value>> {
    info!("Preparing tool call messages");
    debug!("Raw LLM response: {}", response);

    let tool_call = serde_json::from_str::<Value>(response)?;
    if tool_call.get("tool").is_some() && tool_call.get("arguments").is_some() {
        let tool_name = tool_call["tool"]
            .as_str()
            .ok_or_else(|| Error::msg("Invalid tool name"))?;
        let arguments = &tool_call["arguments"];

        info!("Executing tool: {}", tool_name);
        debug!(
            "Tool arguments: {}",
            serde_json::to_string_pretty(arguments)?
        );

        // Format tool response for LLM processing
        let tool_response = serde_json::to_string(&arguments)?;
        Ok(vec![
            json!({
                "role": "system",
                "content": "You are a helpful assistant. Please process the following tool response and provide a clear, natural language explanation of the result. Do not include any technical details or JSON formatting in your response."
            }),
            json!({
                "role": "user",
                "content": format!("Please explain this tool response: {}", tool_response)
            }),
        ])
    } else {
        Err(Error::msg("Invalid tool call format"))
    }
}

/// Main entry point of the application
/// Initializes the chat session and runs the main loop
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging with info level
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .format_module_path(true)
        .format_target(true)
        .init();

    info!("Starting MCP-ChatBot...");

    // Load system prompts
    let prompts = load_system_prompts();

    // Load LLM configuration
    let config = Configuration::new();
    let llm_config = config.load_llm_config("config/config.yaml")?;

    // Check Ollama server status first
    let ollama_status = check_ollama_status().await;
    let current_ai_server = if ollama_status {
        "ollama".to_string()
    } else {
        warn!("Ollama server not available, falling back to Deepseek");
        "deepseek".to_string()
    };

    // Initialize LLM client with appropriate provider
    let llm_client = Arc::new(Mutex::new(LLMClient::new(
        llm_config,
        if current_ai_server == "ollama" {
            LLMProvider::Ollama
        } else {
            LLMProvider::Deepseek
        },
    )));

    let server_status = get_server_status(&current_ai_server, &ollama_status);

    let mut session = ChatSession::new(llm_client).await?;
    session.current_ai_server = current_ai_server.clone();

    // Display welcome message from prompts
    print_colored_ln(&prompts.common_prompts.welcome, Color::Yellow);
    print_colored_ln(
        &format!(
            "Current AI Server: {} ({})",
            server_status, current_ai_server
        ),
        Color::Cyan,
    );
    println!();

    // Start chat session
    session.start().await?;

    // Clean up resources
    session.cleanup_servers().await?;
    info!("MCP Chat Demo stopped");
    Ok(())
}
