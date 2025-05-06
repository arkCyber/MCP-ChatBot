//! Utility Module: Common helper functions for the MCP-ChatBot framework
//!
//! This module provides various utility functions including:
//! - Color formatting for terminal output
//! - Loading animations and visual feedback
//! - Printing helpers with color support
//! - Status checking functions
//!
//! Key Components:
//! - `Color`: Enum for ANSI terminal colors
//! - Print functions for formatted output
//! - Animation functions for visual feedback
//!
//! Author: arkSong <arksong2018@gmail.com>
//! Version: 1.0.0
//! License: MIT


use log::{debug, error, info};
use serde_json::Value;
use std::io::{self, Write};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Copy)]
pub enum Color {
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
}

impl Color {
    pub fn as_ansi_code(&self) -> &str {
        match self {
            Color::Red => "\x1b[31m",
            Color::Green => "\x1b[32m",
            Color::Yellow => "\x1b[33m",
            Color::Blue => "\x1b[34m",
            Color::Magenta => "\x1b[35m",
            Color::Cyan => "\x1b[36m",
            Color::White => "\x1b[37m",
        }
    }
}

pub fn print_colored(text: &str, color: Color) {
    let color_code = color.as_ansi_code();
    print!("{}{}\x1b[0m", color_code, text);
    io::stdout().flush().unwrap();
}

pub fn print_colored_ln(text: &str, color: Color) {
    print_colored(text, color);
    println!();
}

pub fn loading_animation(text: &str, duration: Duration) {
    let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let mut i = 0;
    let start = std::time::Instant::now();

    while start.elapsed() < duration {
        print!("\r{} {} ", frames[i], text);
        io::stdout().flush().unwrap();
        i = (i + 1) % frames.len();
        thread::sleep(Duration::from_millis(100));
    }
    println!();
}

pub async fn print_menu(current_ai_server: &str) {
    print_colored_ln("\nWelcome to MCP-ChatBot Playground!", Color::Yellow);
    print_colored_ln("Your AI Assistant is ready to help.\n", Color::Yellow);

    print_colored_ln("Available Commands:", Color::Cyan);
    print_colored_ln("/help - Show this help menu", Color::White);
    print_colored_ln("/clear - Clear the screen", Color::White);
    print_colored_ln("/usage - Show usage information", Color::White);
    print_colored_ln("/exit - Exit the program", Color::White);
    print_colored_ln("/servers - List available servers", Color::White);
    print_colored_ln("/tools - List available tools", Color::White);
    print_colored_ln("/resources - List available resources", Color::White);
    print_colored_ln("/debug - Toggle debug mode", Color::White);
    print_colored_ln("/ai - Switch between AI servers", Color::White);
    print_colored_ln(
        "/voice - Start voice input (press Enter to stop recording)",
        Color::White,
    );

    print_colored_ln("\nShortcuts:", Color::Cyan);
    print_colored_ln("CTRL+K - Stop current inference", Color::White);
    print_colored_ln("CTRL+C - Exit the program", Color::White);
    print_colored_ln("\nCurrent AI Server: ", Color::Cyan);
    print_colored_ln(current_ai_server, Color::White);
    print_colored_ln("\nType your message or command:", Color::Cyan);
}

pub async fn check_ollama_status() -> bool {
    info!("Starting Ollama server status check...");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .no_proxy()
        .build()
        .unwrap();

    debug!("Checking Ollama server status at http://localhost:11434/api/tags");
    match client.get("http://localhost:11434/api/tags").send().await {
        Ok(response) => {
            info!(
                "Received response from Ollama server with status: {}",
                response.status()
            );
            if response.status().is_success() {
                match response.text().await {
                    Ok(body) => {
                        debug!("Ollama server response body: {}", body);
                        match serde_json::from_str::<serde_json::Value>(&body) {
                            Ok(json) => {
                                debug!("Parsed JSON response: {:?}", json);
                                if let Some(models) = json.get("models") {
                                    if let Some(models_array) = models.as_array() {
                                        let has_model = models_array.iter().any(|model| {
                                            model
                                                .get("name")
                                                .and_then(|n| n.as_str())
                                                .map(|s| s == "llama3.2:latest")
                                                .unwrap_or(false)
                                        });
                                        if has_model {
                                            info!(
                                                "Found llama3.2:latest model in available models"
                                            );
                                            return true;
                                        }
                                    }
                                }
                                error!("llama3.2:latest model not found in available models. Response: {}", body);
                                false
                            }
                            Err(e) => {
                                error!("Failed to parse JSON response: {}", e);
                                false
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to read Ollama response body: {}", e);
                        false
                    }
                }
            } else {
                error!(
                    "Ollama server returned error status: {} with headers: {:?}",
                    response.status(),
                    response.headers()
                );
                false
            }
        }
        Err(e) => {
            error!("Failed to connect to Ollama server: {}", e);
            if e.is_timeout() {
                error!("Connection timed out after 5 seconds");
            } else if e.is_connect() {
                error!("Connection refused - is Ollama server running?");
            }
            false
        }
    }
}

pub fn get_server_status(current_ai_server: &str, ollama_status: &bool) -> String {
    if current_ai_server == "ollama" {
        if *ollama_status {
            "Ollama (running with llama3.2:latest)".to_string()
        } else {
            "Ollama (not running or model not found)".to_string()
        }
    } else if current_ai_server == "openai" {
        "OpenAI (running)".to_string()
    } else {
        format!(
            "Deepseek ({})",
            if current_ai_server == "sk-878a5319c7b14bc48109e19315361" {
                "default"
            } else {
                "custom"
            }
        )
    }
}

pub fn print_about() {
    print_colored_ln("\nMCP Chat Demo", Color::Cyan);
    print_colored_ln("Version: 1.0.0", Color::White);
    print_colored_ln(
        "A demonstration of the Model Context Protocol (MCP)",
        Color::White,
    );
    print_colored_ln("\nFeatures:", Color::Yellow);
    print_colored_ln("- Interactive chat interface", Color::White);
    print_colored_ln("- Tool execution and management", Color::White);
    print_colored_ln("- Resource access and manipulation", Color::White);
    print_colored_ln("- SQLite database integration", Color::White);
    print_colored_ln("- DeepSeek LLM integration", Color::White);
    print_colored_ln("\nAvailable Commands:", Color::Yellow);
    print_colored_ln("/help - Show help menu", Color::White);
    print_colored_ln("/servers - List available servers", Color::White);
    print_colored_ln("/tools - List available tools", Color::White);
    print_colored_ln("/resources - List available resources", Color::White);
    print_colored_ln("/clear - Clear the screen", Color::White);
    print_colored_ln("/usage - Show this information", Color::White);
    print_colored_ln("/exit - Exit the program", Color::White);
    println!();

    print_colored_ln("\nTool Usage Examples:", Color::Yellow);

    print_colored_ln("Memory Server:", Color::Cyan);
    print_colored_ln("  Store a value:", Color::White);
    print_colored_ln("    {\"tool\": \"memory_set\", \"arguments\": {\"key\": \"test\", \"value\": \"hello world\"}}", Color::Green);
    print_colored_ln("  Retrieve a value:", Color::White);
    print_colored_ln(
        "    {\"tool\": \"memory_get\", \"arguments\": {\"key\": \"test\"}}",
        Color::Green,
    );
    println!();

    print_colored_ln("SQLite Server:", Color::Cyan);
    print_colored_ln("  Create a table:", Color::White);
    print_colored_ln("    {\"tool\": \"sqlite_create_table\", \"arguments\": {\"name\": \"users\", \"columns\": [{\"name\": \"id\", \"type\": \"INTEGER\", \"primary_key\": true}, {\"name\": \"name\", \"type\": \"TEXT\"}]}}", Color::Green);
    print_colored_ln("  Execute a query:", Color::White);
    print_colored_ln(
        "    {\"tool\": \"sqlite_query\", \"arguments\": {\"query\": \"SELECT * FROM users\"}}",
        Color::Green,
    );
    println!();

    print_colored_ln("File Server:", Color::Cyan);
    print_colored_ln("  Read a file:", Color::White);
    print_colored_ln(
        "    {\"tool\": \"file_read\", \"arguments\": {\"path\": \"example.txt\"}}",
        Color::Green,
    );
    print_colored_ln("  Write to a file:", Color::White);
    print_colored_ln("    {\"tool\": \"file_write\", \"arguments\": {\"path\": \"example.txt\", \"content\": \"Hello World\"}}", Color::Green);
    println!();

    print_colored_ln("Puppeteer Server:", Color::Cyan);
    print_colored_ln("  Navigate to a URL:", Color::White);
    print_colored_ln(
        "    {\"tool\": \"puppeteer_navigate\", \"arguments\": {\"url\": \"https://example.com\"}}",
        Color::Green,
    );
    print_colored_ln("  Click an element:", Color::White);
    print_colored_ln(
        "    {\"tool\": \"puppeteer_click\", \"arguments\": {\"selector\": \"button.submit\"}}",
        Color::Green,
    );
    println!();

    print_colored_ln("\nReal Conversation Examples:", Color::Yellow);

    print_colored_ln("Example 1 - Memory Operations:", Color::Cyan);
    print_colored_ln("  User: Store my name as 'John'", Color::White);
    print_colored_ln("  Bot: {\"tool\": \"memory_set\", \"arguments\": {\"key\": \"user_name\", \"value\": \"John\"}}", Color::Green);
    print_colored_ln("  User: What's my name?", Color::White);
    print_colored_ln(
        "  Bot: {\"tool\": \"memory_get\", \"arguments\": {\"key\": \"user_name\"}}",
        Color::Green,
    );
    print_colored_ln("  Bot: Your name is John", Color::White);
    println!();

    print_colored_ln("Example 2 - SQLite Database:", Color::Cyan);
    print_colored_ln(
        "  User: Create a table for storing user information",
        Color::White,
    );
    print_colored_ln("  Bot: {\"tool\": \"sqlite_create_table\", \"arguments\": {\"name\": \"users\", \"columns\": [{\"name\": \"id\", \"type\": \"INTEGER\", \"primary_key\": true}, {\"name\": \"name\", \"type\": \"TEXT\"}, {\"name\": \"email\", \"type\": \"TEXT\"}]}}", Color::Green);
    print_colored_ln("  User: Show me all users", Color::White);
    print_colored_ln(
        "  Bot: {\"tool\": \"sqlite_query\", \"arguments\": {\"query\": \"SELECT * FROM users\"}}",
        Color::Green,
    );
    println!();

    print_colored_ln("Example 3 - File Operations:", Color::Cyan);
    print_colored_ln("  User: Create a new file with some content", Color::White);
    print_colored_ln("  Bot: {\"tool\": \"file_write\", \"arguments\": {\"path\": \"notes.txt\", \"content\": \"This is a test note.\"}}", Color::Green);
    print_colored_ln("  User: Read the file content", Color::White);
    print_colored_ln(
        "  Bot: {\"tool\": \"file_read\", \"arguments\": {\"path\": \"notes.txt\"}}",
        Color::Green,
    );
    println!();

    print_colored_ln("Example 4 - Web Automation:", Color::Cyan);
    print_colored_ln(
        "  User: Go to example.com and click the login button",
        Color::White,
    );
    print_colored_ln("  Bot: {\"tool\": \"puppeteer_navigate\", \"arguments\": {\"url\": \"https://example.com\"}}", Color::Green);
    print_colored_ln(
        "  Bot: {\"tool\": \"puppeteer_click\", \"arguments\": {\"selector\": \"button.login\"}}",
        Color::Green,
    );
    println!();
}

pub fn print_servers(config: &serde_json::Value) {
    print_colored_ln("\nAvailable Servers:", Color::Cyan);
    if let Some(servers) = config.get("mcpServers") {
        for (name, _) in servers.as_object().unwrap() {
            print_colored_ln(&format!("- {}", name), Color::White);
        }
    }
    println!();
}

pub fn print_resources(resources: &[crate::protocol::ResourceSchema]) {
    if resources.is_empty() {
        print_colored_ln("  No resources available", Color::White);
        return;
    }

    print_colored_ln("\nAvailable Resources:", Color::Cyan);
    for resource in resources {
        print_colored_ln(&format!("\nResource: {}", resource.pattern), Color::Yellow);
        print_colored_ln(
            &format!("  Description: {}", resource.description),
            Color::White,
        );

        // Display input parameters
        if let Some(input_schema) = &resource.input_schema {
            if let Some(properties) = input_schema.get("properties") {
                if let Some(properties) = properties.as_object() {
                    print_colored_ln("  Parameters:", Color::White);
                    for (param_name, param_info) in properties {
                        let mut param_desc = format!("    - {}: ", param_name);
                        if let Some(desc) = param_info.get("description") {
                            if let Some(desc) = desc.as_str() {
                                param_desc.push_str(desc);
                            }
                        }
                        if let Some(required) = input_schema.get("required") {
                            if let Some(required) = required.as_array() {
                                if required.contains(&Value::String(param_name.clone())) {
                                    param_desc.push_str(" (required)");
                                }
                            }
                        }
                        print_colored_ln(&param_desc, Color::White);
                    }
                }
            }
        }

        // Display output fields
        if let Some(output_schema) = &resource.output_schema {
            if let Some(properties) = output_schema.get("properties") {
                if let Some(properties) = properties.as_object() {
                    print_colored_ln("  Returns:", Color::White);
                    for (field_name, field_info) in properties {
                        let mut field_desc = format!("    - {}: ", field_name);
                        if let Some(desc) = field_info.get("description") {
                            if let Some(desc) = desc.as_str() {
                                field_desc.push_str(desc);
                            }
                        }
                        print_colored_ln(&field_desc, Color::White);
                    }
                }
            }
        }
    }
    println!();
}

pub fn print_mcp_servers(config: &Value) {
    print_colored_ln("\nAvailable MCP Servers:", Color::Cyan);
    if let Some(servers) = config.get("mcpServers") {
        for (name, server) in servers.as_object().unwrap() {
            print_colored(&format!("- {}: ", name), Color::Yellow);
            if let Some(command) = server.get("command") {
                print_colored(&format!("{} ", command.as_str().unwrap()), Color::Green);
            }
            if let Some(args) = server.get("args") {
                let args_str = args
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|arg| arg.as_str().unwrap())
                    .collect::<Vec<_>>()
                    .join(" ");
                print_colored(&format!("{}", args_str), Color::White);
            }
            println!();
        }
    }
    println!();
}

pub fn print_tools(tools: &[crate::protocol::ToolSchema]) {
    if tools.is_empty() {
        print_colored_ln("  No tools available", Color::White);
        return;
    }

    for tool in tools {
        print_colored_ln(&format!("\n  Tool: {}", tool.name), Color::Yellow);
        print_colored_ln(
            &format!("    Description: {}", tool.description),
            Color::White,
        );

        // Display input parameters
        if let Some(properties) = tool.input_schema.get("properties") {
            if let Some(properties) = properties.as_object() {
                print_colored_ln("    Parameters:", Color::White);
                for (param_name, param_info) in properties {
                    let mut param_desc = format!("      - {}: ", param_name);
                    if let Some(desc) = param_info.get("description") {
                        if let Some(desc) = desc.as_str() {
                            param_desc.push_str(desc);
                        }
                    }
                    if let Some(required) = tool.input_schema.get("required") {
                        if let Some(required) = required.as_array() {
                            if required.contains(&Value::String(param_name.clone())) {
                                param_desc.push_str(" (required)");
                            }
                        }
                    }
                    print_colored_ln(&param_desc, Color::White);
                }
            }
        }

        // Display output fields
        if let Some(output_schema) = &tool.output_schema {
            if let Some(properties) = output_schema.get("properties") {
                if let Some(properties) = properties.as_object() {
                    print_colored_ln("    Returns:", Color::White);
                    for (field_name, field_info) in properties {
                        let mut field_desc = format!("      - {}: ", field_name);
                        if let Some(desc) = field_info.get("description") {
                            if let Some(desc) = desc.as_str() {
                                field_desc.push_str(desc);
                            }
                        }
                        print_colored_ln(&field_desc, Color::White);
                    }
                }
            }
        }
    }
    println!();
}

pub fn typing_animation(text: &str, delay_ms: u64) {
    for c in text.chars() {
        print!("{}", c);
        io::stdout().flush().unwrap();
        thread::sleep(Duration::from_millis(delay_ms));
    }
    println!();
}

pub fn print_download_animation(text: &str, duration: Duration) {
    let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let mut i = 0;
    let start = std::time::Instant::now();
    let mut progress = 0;
    let total = 100;

    while start.elapsed() < duration {
        progress = (progress + 1) % (total + 1);
        let percentage = progress;
        print!("\r{} {} [{}%] ", frames[i], text, percentage);
        io::stdout().flush().unwrap();
        i = (i + 1) % frames.len();
        thread::sleep(Duration::from_millis(50));
    }
    println!();
}

pub fn print_download_progress(text: &str, current: usize, total: usize) {
    let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let i = (current % frames.len()) as usize;
    let percentage = (current as f64 / total as f64 * 100.0) as usize;
    print!(
        "\r{} {} [{}/{}] [{}%] ",
        frames[i], text, current, total, percentage
    );
    io::stdout().flush().unwrap();
}

pub fn print_waiting_animation(text: &str, duration: Duration) {
    let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let dots = ["", ".", "..", "..."];
    let mut i = 0;
    let mut dot_index = 0;
    let start = std::time::Instant::now();

    while start.elapsed() < duration {
        print!("\r{} {} {}", frames[i], text, dots[dot_index]);
        io::stdout().flush().unwrap();
        i = (i + 1) % frames.len();
        dot_index = (dot_index + 1) % dots.len();
        thread::sleep(Duration::from_millis(100));
    }
    println!();
}

pub fn print_bot_thinking(text: &str) {
    let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let dots = ["", ".", "..", "..."];
    let mut i = 0;
    let mut dot_index = 0;

    print!("\r{} {} {}", frames[i], text, dots[dot_index]);
    io::stdout().flush().unwrap();
    i = (i + 1) % frames.len();
    dot_index = (dot_index + 1) % dots.len();
}

pub async fn print_bot_thinking_continuous(stop_signal: Arc<Mutex<bool>>) {
    let frames = vec!["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let dots = vec!["", ".", "..", "..."];
    let mut i = 0;
    let mut dot_index = 0;

    loop {
        if *stop_signal.lock().await {
            // 清理当前行
            print!("\r\x1b[K");
            io::stdout().flush().unwrap();
            break;
        }

        print!("\rBOT: Thinking{} {}", frames[i], dots[dot_index]);
        io::stdout().flush().unwrap();

        i = (i + 1) % frames.len();
        dot_index = (dot_index + 1) % dots.len();

        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

pub async fn print_recording_animation(stop_signal: Arc<Mutex<bool>>) {
    let frames = vec!["🎤", "🎙️", "🎧", "🎵"];
    let dots = vec!["", ".", "..", "..."];
    let mut i = 0;
    let mut dot_index = 0;

    loop {
        if *stop_signal.lock().await {
            print!("\r\x1b[K");
            io::stdout().flush().unwrap();
            break;
        }

        print!("\rRecording{} {}", frames[i], dots[dot_index]);
        io::stdout().flush().unwrap();

        i = (i + 1) % frames.len();
        dot_index = (dot_index + 1) % dots.len();

        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}
