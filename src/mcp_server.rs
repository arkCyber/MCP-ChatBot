//! MCP Server Module: Core server functionality for the MCP-ChatBot framework
//!
//! This module implements the main server functionality including:
//! - Tool registration and execution
//! - Resource management
//! - Prompt handling
//! - Retry mechanisms for tool execution
//! - Progress tracking
//!
//! Author: arkSong <arksong2018@gmail.com>
//! Version: 1.0.0
//! License: MIT

use anyhow::{Error, Result};
use log::{info, warn};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};

use crate::protocol::{PromptArgument, PromptSchema, ResourceSchema, ToolSchema};

/// Represents a tool that can be registered and executed by the MCP server
pub struct Tool {
    /// Name of the tool
    pub name: String,
    /// Description of what the tool does
    pub description: String,
    /// JSON schema defining the expected input format
    pub input_schema: Value,
    /// Optional JSON schema defining the output format
    pub output_schema: Option<Value>,
    /// Async function that implements the tool's functionality
    handler: Box<
        dyn Fn(
                Value,
            )
                -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value>> + Send + Sync>>
            + Send
            + Sync,
    >,
}

impl Tool {
    /// Creates a new Tool instance
    pub fn new(
        name: String,
        description: String,
        input_schema: Value,
        output_schema: Option<Value>,
        handler: Box<
            dyn Fn(
                    Value,
                ) -> std::pin::Pin<
                    Box<dyn std::future::Future<Output = Result<Value>> + Send + Sync>,
                > + Send
                + Sync,
        >,
    ) -> Self {
        Self {
            name,
            description,
            input_schema,
            output_schema,
            handler,
        }
    }

    /// Returns the tool's schema information
    pub fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
            output_schema: self.output_schema.clone(),
        }
    }

    /// Executes the tool with the given arguments
    pub async fn execute(&self, arguments: &Value) -> Result<Value> {
        (self.handler)(arguments.clone()).await
    }

    /// Formats the tool information for LLM consumption
    pub fn format_for_llm(&self) -> String {
        let mut args_desc = Vec::new();
        if let Some(properties) = self.input_schema.get("properties") {
            if let Some(properties) = properties.as_object() {
                for (param_name, param_info) in properties {
                    let mut arg_desc = format!("- {}: ", param_name);
                    if let Some(desc) = param_info.get("description") {
                        if let Some(desc) = desc.as_str() {
                            arg_desc.push_str(desc);
                        }
                    } else {
                        arg_desc.push_str("No description");
                    }

                    if let Some(required) = self.input_schema.get("required") {
                        if let Some(required) = required.as_array() {
                            if required.contains(&Value::String(param_name.clone())) {
                                arg_desc.push_str(" (required)");
                            }
                        }
                    }
                    args_desc.push(arg_desc);
                }
            }
        }

        format!(
            "\nTool: {}\nDescription: {}\nArguments:\n{}",
            self.name,
            self.description,
            args_desc.join("\n")
        )
    }
}

/// Represents a resource that can be accessed by the MCP server
pub struct Resource {
    /// Pattern used to match resource requests
    pub pattern: String,
    /// Description of the resource
    pub description: String,
    /// Optional JSON schema for input parameters
    pub input_schema: Option<Value>,
    /// Optional JSON schema defining the output format
    pub output_schema: Option<Value>,
    /// Async function that implements resource access
    handler: Box<
        dyn Fn(
                Value,
            )
                -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value>> + Send + Sync>>
            + Send
            + Sync,
    >,
}

impl Resource {
    /// Creates a new Resource instance
    pub fn new(
        pattern: String,
        description: String,
        input_schema: Option<Value>,
        output_schema: Option<Value>,
        handler: Box<
            dyn Fn(
                    Value,
                ) -> std::pin::Pin<
                    Box<dyn std::future::Future<Output = Result<Value>> + Send + Sync>,
                > + Send
                + Sync,
        >,
    ) -> Self {
        Self {
            pattern,
            description,
            input_schema,
            output_schema,
            handler,
        }
    }

    /// Returns the resource's schema information
    pub fn schema(&self) -> ResourceSchema {
        ResourceSchema {
            pattern: self.pattern.clone(),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
            output_schema: self.output_schema.clone(),
        }
    }

    /// Reads from the resource with the given arguments
    pub async fn read(&self, arguments: &Value) -> Result<Value> {
        (self.handler)(arguments.clone()).await
    }
}

/// Represents a prompt template that can be used by the MCP server
pub struct Prompt {
    /// Name of the prompt
    pub name: String,
    /// Description of the prompt's purpose
    pub description: String,
    /// List of arguments the prompt accepts
    pub arguments: Vec<PromptArgument>,
    /// Function that generates the prompt text
    handler: Box<dyn Fn(Option<Value>) -> Result<Value> + Send + Sync>,
}

impl Prompt {
    /// Creates a new Prompt instance
    pub fn new(
        name: String,
        description: String,
        arguments: Vec<PromptArgument>,
        handler: Box<dyn Fn(Option<Value>) -> Result<Value> + Send + Sync>,
    ) -> Self {
        Self {
            name,
            description,
            arguments,
            handler,
        }
    }

    /// Returns the prompt's schema information
    pub fn schema(&self) -> PromptSchema {
        PromptSchema {
            name: self.name.clone(),
            description: self.description.clone(),
            arguments: self.arguments.clone(),
        }
    }

    /// Generates the prompt text with optional arguments
    pub async fn get(&self, arguments: Option<Value>) -> Result<Value> {
        (self.handler)(arguments)
    }
}

/// Main MCP server struct that manages tools, resources and prompts
#[derive(Clone)]
pub struct McpServer {
    /// Thread-safe collection of registered tools
    tools: Arc<Mutex<Vec<Tool>>>,
    /// Thread-safe collection of registered resources
    resources: Arc<Mutex<Vec<Resource>>>,
    /// Thread-safe collection of registered prompts
    prompts: Arc<Mutex<Vec<Prompt>>>,
}

impl McpServer {
    /// Creates a new McpServer instance
    pub fn new() -> Self {
        Self {
            tools: Arc::new(Mutex::new(Vec::new())),
            resources: Arc::new(Mutex::new(Vec::new())),
            prompts: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Initializes the server
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing MCP server");
        Ok(())
    }

    /// Cleans up server resources
    pub async fn cleanup(&mut self) -> Result<()> {
        info!("Cleaning up MCP server");
        self.tools.lock().await.clear();
        self.resources.lock().await.clear();
        self.prompts.lock().await.clear();
        Ok(())
    }

    /// Registers a new tool with the server
    pub async fn register_tool(&mut self, tool: Tool) -> Result<()> {
        self.tools.lock().await.push(tool);
        Ok(())
    }

    /// Registers a new resource with the server
    pub async fn register_resource(&mut self, resource: Resource) -> Result<()> {
        self.resources.lock().await.push(resource);
        Ok(())
    }

    /// Registers a new prompt with the server
    pub async fn register_prompt(&mut self, prompt: Prompt) -> Result<()> {
        self.prompts.lock().await.push(prompt);
        Ok(())
    }

    /// Lists all registered tools
    pub async fn list_tools(&self) -> Result<Vec<ToolSchema>> {
        Ok(self.tools.lock().await.iter().map(|t| t.schema()).collect())
    }

    /// Executes a tool with retry mechanism
    pub async fn execute_tool(
        &mut self,
        name: &str,
        arguments: &Value,
        retries: u32,
        delay: Duration,
    ) -> Result<Value> {
        let mut attempt = 0;
        while attempt < retries {
            match self.tools.lock().await.iter().find(|t| t.name == *name) {
                Some(tool) => {
                    match tool.execute(arguments).await {
                        Ok(result) => {
                            // Handle progress information
                            if let Some(progress) = result.get("progress") {
                                if let Some(total) = result.get("total") {
                                    let progress = progress.as_f64().unwrap_or(0.0);
                                    let total = total.as_f64().unwrap_or(1.0);
                                    let percentage = (progress / total) * 100.0;
                                    info!("Progress: {}/{} ({:.1}%)", progress, total, percentage);
                                }
                            }
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
                                return Err(e);
                            }
                        }
                    }
                }
                None => return Err(Error::msg(format!("Tool not found: {}", name))),
            }
        }
        Err(Error::msg("Max retries reached"))
    }

    /// Lists all registered resources
    pub async fn list_resources(&self) -> Result<Vec<ResourceSchema>> {
        Ok(self
            .resources
            .lock()
            .await
            .iter()
            .map(|r| r.schema())
            .collect())
    }

    /// Reads from a resource with the given pattern and arguments
    pub async fn read_resource(&mut self, pattern: &str, arguments: &Value) -> Result<Value> {
        match self
            .resources
            .lock()
            .await
            .iter()
            .find(|r| r.pattern == *pattern)
        {
            Some(resource) => resource.read(arguments).await,
            None => Err(Error::msg(format!("Resource not found: {}", pattern))),
        }
    }

    /// Lists all registered prompts
    pub async fn list_prompts(&self) -> Result<Vec<PromptSchema>> {
        Ok(self
            .prompts
            .lock()
            .await
            .iter()
            .map(|p| p.schema())
            .collect())
    }

    /// Gets a prompt with optional arguments
    pub async fn get_prompt(&mut self, name: &str, arguments: Option<Value>) -> Result<Value> {
        match self.prompts.lock().await.iter().find(|p| p.name == *name) {
            Some(prompt) => prompt.get(arguments).await,
            None => Err(Error::msg(format!("Prompt not found: {}", name))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Duration;

    async fn create_test_tool() -> Tool {
        Tool::new(
            "test_tool".to_string(),
            "A test tool".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Test input"
                    }
                },
                "required": ["input"]
            }),
            Some(json!({
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string"
                    }
                }
            })),
            Box::new(|args| {
                Box::pin(async move {
                    let input = args["input"].as_str().unwrap_or("default");
                    Ok(json!({ "result": format!("Processed: {}", input) }))
                })
            }),
        )
    }

    async fn create_test_resource() -> Resource {
        Resource::new(
            "test_resource".to_string(),
            "A test resource".to_string(),
            Some(json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Test query"
                    }
                }
            })),
            Some(json!({
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string"
                    }
                }
            })),
            Box::new(|args| {
                Box::pin(async move {
                    let query = args["query"].as_str().unwrap_or("default");
                    Ok(json!({ "data": format!("Resource data for: {}", query) }))
                })
            }),
        )
    }

    async fn create_test_prompt() -> Prompt {
        Prompt::new(
            "test_prompt".to_string(),
            "A test prompt".to_string(),
            vec![PromptArgument {
                name: "context".to_string(),
                description: "Test context".to_string(),
                required: true,
                schema: json!({
                    "type": "string",
                    "description": "Test context"
                }),
            }],
            Box::new(|args| {
                Ok(json!({
                    "text": format!(
                        "Test prompt with context: {}",
                        args.and_then(|v| v["context"].as_str().map(|s| s.to_string()))
                            .unwrap_or_else(|| "default".to_string())
                    )
                }))
            }),
        )
    }

    #[tokio::test]
    async fn test_server_initialization() {
        let mut server = McpServer::new();
        assert!(server.initialize().await.is_ok());
        assert!(server.cleanup().await.is_ok());
    }

    #[tokio::test]
    async fn test_tool_registration_and_execution() {
        let mut server = McpServer::new();
        let tool = create_test_tool().await;

        // Register tool
        assert!(server.register_tool(tool).await.is_ok());

        // List tools
        let tools = server.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "test_tool");

        // Execute tool
        let result = server
            .execute_tool(
                "test_tool",
                &json!({ "input": "test" }),
                3,
                Duration::from_millis(100),
            )
            .await
            .unwrap();
        assert_eq!(result["result"], "Processed: test");
    }

    #[tokio::test]
    async fn test_resource_registration_and_access() {
        let mut server = McpServer::new();
        let resource = create_test_resource().await;

        // Register resource
        assert!(server.register_resource(resource).await.is_ok());

        // List resources
        let resources = server.list_resources().await.unwrap();
        assert_eq!(resources.len(), 1);
        assert_eq!(resources[0].pattern, "test_resource");

        // Read resource
        let result = server
            .read_resource("test_resource", &json!({ "query": "test" }))
            .await
            .unwrap();
        assert_eq!(result["data"], "Resource data for: test");
    }

    #[tokio::test]
    async fn test_prompt_registration_and_generation() {
        let mut server = McpServer::new();
        let prompt = create_test_prompt().await;

        // Register prompt
        assert!(server.register_prompt(prompt).await.is_ok());

        // List prompts
        let prompts = server.list_prompts().await.unwrap();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].name, "test_prompt");

        // Get prompt
        let result = server
            .get_prompt("test_prompt", Some(json!({ "context": "test context" })))
            .await
            .unwrap();
        assert_eq!(result["text"], "Test prompt with context: test context");
    }

    #[tokio::test]
    async fn test_tool_execution_retry() {
        let mut server = McpServer::new();
        let attempts = Arc::new(Mutex::new(0));
        let attempts_clone = attempts.clone();

        let tool = Tool::new(
            "failing_tool".to_string(),
            "A tool that fails twice then succeeds".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Test input"
                    }
                }
            }),
            None,
            Box::new(move |_| {
                let attempts = attempts_clone.clone();
                Box::pin(async move {
                    let mut count = attempts.lock().await;
                    *count += 1;
                    if *count < 3 {
                        Err(Error::msg("Temporary failure"))
                    } else {
                        Ok(json!({ "result": "success" }))
                    }
                })
            }),
        );

        server.register_tool(tool).await.unwrap();

        let result = server
            .execute_tool(
                "failing_tool",
                &json!({ "input": "test" }),
                3,
                Duration::from_millis(100),
            )
            .await
            .unwrap();
        assert_eq!(result["result"], "success");
    }

    #[tokio::test]
    async fn test_error_handling() {
        let mut server = McpServer::new();

        // Test non-existent tool
        let result = server
            .execute_tool(
                "non_existent_tool",
                &json!({}),
                1,
                Duration::from_millis(100),
            )
            .await;
        assert!(result.is_err());

        // Test non-existent resource
        let result = server
            .read_resource("non_existent_resource", &json!({}))
            .await;
        assert!(result.is_err());

        // Test non-existent prompt
        let result = server.get_prompt("non_existent_prompt", None).await;
        assert!(result.is_err());
    }
}
