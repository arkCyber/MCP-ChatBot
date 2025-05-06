#[cfg(test)]
mod tests {
    use anyhow::Result;
    use mcp_chatbot::{
        llm_client::{LLMClient, LLMProvider},
        load_system_prompts, ChatSession, Configuration, Message, Server, Tool,
    };
    use serde_json::json;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    #[test]
    fn test_configuration_new() {
        // Save current API key
        let api_key = std::env::var("LLM_API_KEY").ok();
        // Remove API key for test
        std::env::remove_var("LLM_API_KEY");

        let config = Configuration::new();
        assert!(config.api_key.is_none());

        // Restore API key if it existed
        if let Some(key) = api_key {
            std::env::set_var("LLM_API_KEY", key);
        }
    }

    #[test]
    fn test_message_creation() {
        let message = Message::ExecuteTool {
            name: "test_tool".to_string(),
            arguments: json!({"content": "Hello, world!"}),
        };
        match message {
            Message::ExecuteTool { name, arguments } => {
                assert_eq!(name, "test_tool");
                assert_eq!(arguments["content"], "Hello, world!");
            }
            _ => panic!("Expected ExecuteTool message"),
        }
    }

    #[test]
    fn test_llm_client_initialization() {
        let config = Configuration::new()
            .load_llm_config("config/config.yaml")
            .unwrap();
        let client = LLMClient::new(config, LLMProvider::Ollama);
        assert!(true);
    }

    #[tokio::test]
    async fn test_chat_session_initialization() -> Result<()> {
        let config = Configuration::new()
            .load_llm_config("config/config.yaml")
            .unwrap();
        let llm_client = Arc::new(Mutex::new(LLMClient::new(config, LLMProvider::Ollama)));
        let session = ChatSession::new(llm_client).await?;
        assert_eq!(session.current_provider, LLMProvider::Ollama);
        assert!(session.running);
        Ok(())
    }

    #[test]
    fn test_system_prompts_loading() {
        let prompts = load_system_prompts();
        assert!(!prompts.default_system_prompt.is_empty());
        assert!(!prompts.common_prompts.welcome.is_empty());
        assert!(!prompts.common_prompts.tool_response.is_empty());
    }

    #[tokio::test]
    async fn test_command_processing() -> Result<()> {
        let config = Configuration::new()
            .load_llm_config("config/config.yaml")
            .unwrap();
        let llm_client = Arc::new(Mutex::new(LLMClient::new(config, LLMProvider::Ollama)));
        let mut session = ChatSession::new(llm_client).await?;

        // Test help command
        session.process_command("/help").await?;

        // Test clear command
        session.process_command("/clear").await?;

        // Test exit command
        session.process_command("/exit").await?;
        assert!(!session.running);

        Ok(())
    }

    #[tokio::test]
    async fn test_server_initialization() -> Result<()> {
        let config = Configuration::new();
        let server_config = config.load_config("config/servers_config.json")?;
        assert!(server_config.get("mcpServers").is_some());
        Ok(())
    }

    #[tokio::test]
    async fn test_tool_execution() -> Result<()> {
        let config = Configuration::new()
            .load_llm_config("config/config.yaml")
            .unwrap();
        let llm_client = Arc::new(Mutex::new(LLMClient::new(config, LLMProvider::Ollama)));
        let mut session = ChatSession::new(llm_client).await?;

        // Create and initialize memory server with tools
        let memory_config = json!({
            "command": "memory",
            "args": [],
            "env": {},
            "tools": [{
                "name": "memory_set",
                "description": "Set a value in memory",
                "parameters": {
                    "key": "string",
                    "value": "string"
                }
            }]
        });
        let mut memory_server = Server::new("memory".to_string(), memory_config).await?;

        // Create and register the memory_set tool
        let memory_set_tool = Tool::new(
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
            Box::new(|args| Box::pin(async move { Ok(json!({"success": true})) })),
        );

        memory_server
            .mcp_server
            .register_tool(memory_set_tool)
            .await?;
        session.servers.push(memory_server);

        // Test memory tool
        let memory_server = session
            .servers
            .iter_mut()
            .find(|s| s.name == "memory")
            .ok_or_else(|| anyhow::anyhow!("Memory server not found"))?;

        let result = memory_server
            .execute_tool(
                "memory_set",
                &json!({"key": "test", "value": "value"}),
                3,
                std::time::Duration::from_secs(1),
            )
            .await?;

        assert!(result["success"].as_bool().unwrap_or(false));
        Ok(())
    }
}
