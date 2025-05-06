//! Standard Input/Output (stdio) server implementation for MCP chatbot.
//! This module provides a simple command-line interface for interacting with the MCP server.
//! It handles user input through stdin and outputs responses through stdout.
//! This module provides:
//! - Command-line interface for user interaction
//! - Input handling through stdin
//! - Output formatting to stdout
//! - Message history management
//! - Integration with MCP server and LLM client
//!
//! Key Components:
//! - `StdioServer`: Main server struct managing I/O operations
//! - Input/output stream handling
//! - Message buffering and history
//! - Server state management
//!
//! Author: arkSong <arksong2018@gmail.com>
//! Version: 1.0.0
//! License: MIT

use std::io;
use std::sync::Arc;

use anyhow::{Error, Result};
use log::info;
use serde_json::Value;
use tokio::sync::Mutex;

use crate::llm_client::LLMClient;
use crate::mcp_server::McpServer;
use crate::utils::{print_menu, Color};

/// Standard Input/Output server for MCP chatbot.
/// This server provides a command-line interface for interacting with the MCP server.
/// It handles user input through stdin and outputs responses through stdout.
pub struct StdioServer {
    /// Reference to the MCP server instance, wrapped in Arc<Mutex> for thread safety
    server: Arc<Mutex<McpServer>>,
    /// Reference to the LLM client instance, wrapped in Arc<Mutex> for thread safety
    llm_client: Arc<Mutex<LLMClient>>,
    /// Standard input stream for reading user input
    stdin: io::Stdin,
    /// Standard output stream for writing responses
    stdout: io::Stdout,
    /// Flag indicating whether the server is running
    running: bool,
    /// Buffer for storing message history
    messages: Vec<Value>,
}

impl StdioServer {
    /// Creates a new StdioServer instance.
    ///
    /// # Arguments
    /// * `server` - A thread-safe reference to the MCP server
    /// * `llm_client` - A thread-safe reference to the LLM client
    ///
    /// # Returns
    /// A new StdioServer instance
    pub fn new(server: Arc<Mutex<McpServer>>, llm_client: Arc<Mutex<LLMClient>>) -> Self {
        Self {
            server,
            llm_client,
            stdin: io::stdin(),
            stdout: io::stdout(),
            running: true,
            messages: Vec::new(),
        }
    }

    /// Starts the stdio server and begins processing user input.
    ///
    /// This method runs in a loop, reading input from stdin and processing it
    /// until the user enters "/exit" or an error occurs.
    ///
    /// # Returns
    /// * `Result<()>` - Ok if the server runs successfully, Err if an error occurs
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting stdio server");
        while self.running {
            let mut input = String::new();
            self.stdin.read_line(&mut input)?;
            if input.trim() == "/exit" {
                self.running = false;
                break;
            }
            self.process_input(&input).await?;
        }
        Ok(())
    }

    /// Processes user input and generates appropriate responses.
    ///
    /// This method is responsible for:
    /// 1. Parsing the user input
    /// 2. Determining if it's a command or a message
    /// 3. Executing the appropriate action
    /// 4. Generating and sending the response
    ///
    /// # Arguments
    /// * `input` - The user's input string
    ///
    /// # Returns
    /// * `Result<()>` - Ok if processing succeeds, Err if an error occurs
    async fn process_input(&mut self, _input: &str) -> Result<(), Error> {
        // TODO: Implement input processing logic
        // 1. Check if input is a command (starts with '/')
        // 2. If it's a command, handle it appropriately
        // 3. If it's a message, send it to the LLM client
        // 4. Format and display the response
        Ok(())
    }
}
