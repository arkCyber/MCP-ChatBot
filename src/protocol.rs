//! MCP-ChatBot Protocol: Communication protocol for the MCP-ChatBot framework
//!
//! This module defines the communication protocol used by the MCP-ChatBot framework,
//! including messages for initialization, tool execution, resource management,
//! prompt handling, and lifecycle management.
//!
//! Author: arkSong <arksong2018@gmail.com>
//! Version: 1.0.0
//! License: MIT

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize, Deserialize)]
pub enum Message {
    // Initialization messages
    Initialize {
        server_name: String,
        server_version: String,
        capabilities: Capabilities,
    },
    InitializeResponse {
        success: bool,
        error: Option<String>,
    },

    // Tool related messages
    ListTools,
    ListToolsResponse {
        tools: Vec<ToolSchema>,
    },
    ExecuteTool {
        name: String,
        arguments: Value,
    },
    ExecuteToolResponse {
        result: Value,
        error: Option<String>,
    },

    // Resource related messages
    ListResources,
    ListResourcesResponse {
        resources: Vec<ResourceSchema>,
    },
    ReadResource {
        pattern: String,
        arguments: Value,
    },
    ReadResourceResponse {
        result: Value,
        error: Option<String>,
    },

    // Prompt related messages
    ListPrompts,
    ListPromptsResponse {
        prompts: Vec<PromptSchema>,
    },
    GetPrompt {
        name: String,
        arguments: Option<Value>,
    },
    GetPromptResponse {
        result: Value,
        error: Option<String>,
    },

    // Lifecycle messages
    Shutdown,
    ShutdownResponse {
        success: bool,
        error: Option<String>,
    },

    // Error messages
    Error {
        message: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Capabilities {
    pub tools: bool,
    pub resources: bool,
    pub prompts: bool,
    pub notifications: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub output_schema: Option<Value>,
}

impl ToolSchema {
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResourceSchema {
    pub pattern: String,
    pub description: String,
    pub input_schema: Option<Value>,
    pub output_schema: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PromptSchema {
    pub name: String,
    pub description: String,
    pub arguments: Vec<PromptArgument>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PromptArgument {
    pub name: String,
    pub description: String,
    pub required: bool,
    pub schema: Value,
}

// Message serialization and deserialization
impl Message {
    pub fn serialize(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    pub fn deserialize(data: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_message_serialization_deserialization() {
        // Test Initialize message
        let init_msg = Message::Initialize {
            server_name: "test_server".to_string(),
            server_version: "1.0.0".to_string(),
            capabilities: Capabilities {
                tools: true,
                resources: true,
                prompts: true,
                notifications: false,
            },
        };
        let serialized = init_msg.serialize();
        let deserialized = Message::deserialize(&serialized).unwrap();
        assert!(matches!(deserialized, Message::Initialize { .. }));

        // Test ExecuteTool message
        let tool_msg = Message::ExecuteTool {
            name: "test_tool".to_string(),
            arguments: json!({"param1": "value1"}),
        };
        let serialized = tool_msg.serialize();
        let deserialized = Message::deserialize(&serialized).unwrap();
        assert!(matches!(deserialized, Message::ExecuteTool { .. }));

        // Test Error message
        let error_msg = Message::Error {
            message: "test error".to_string(),
        };
        let serialized = error_msg.serialize();
        let deserialized = Message::deserialize(&serialized).unwrap();
        assert!(matches!(deserialized, Message::Error { .. }));
    }

    #[test]
    fn test_tool_schema_formatting() {
        let schema = ToolSchema {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: json!({
                "properties": {
                    "param1": {
                        "description": "First parameter",
                        "type": "string"
                    },
                    "param2": {
                        "description": "Second parameter",
                        "type": "number"
                    }
                },
                "required": ["param1"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string"
                    }
                }
            })),
        };

        let formatted = schema.format_for_llm();
        assert!(formatted.contains("test_tool"));
        assert!(formatted.contains("A test tool"));
        assert!(formatted.contains("param1"));
        assert!(formatted.contains("First parameter"));
        assert!(formatted.contains("(required)"));
        assert!(formatted.contains("param2"));
        assert!(formatted.contains("Second parameter"));
    }

    #[test]
    fn test_resource_schema() {
        let schema = ResourceSchema {
            pattern: "test/*.txt".to_string(),
            description: "Test resource pattern".to_string(),
            input_schema: Some(json!({
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string"
                    }
                }
            })),
            output_schema: Some(json!({
                "type": "array",
                "items": {
                    "type": "string"
                }
            })),
        };

        assert_eq!(schema.pattern, "test/*.txt");
        assert_eq!(schema.description, "Test resource pattern");
        assert!(schema.input_schema.is_some());
        assert!(schema.output_schema.is_some());
    }

    #[test]
    fn test_prompt_schema() {
        let schema = PromptSchema {
            name: "test_prompt".to_string(),
            description: "A test prompt".to_string(),
            arguments: vec![
                PromptArgument {
                    name: "arg1".to_string(),
                    description: "First argument".to_string(),
                    required: true,
                    schema: json!({"type": "string"}),
                },
                PromptArgument {
                    name: "arg2".to_string(),
                    description: "Second argument".to_string(),
                    required: false,
                    schema: json!({"type": "number"}),
                },
            ],
        };

        assert_eq!(schema.name, "test_prompt");
        assert_eq!(schema.description, "A test prompt");
        assert_eq!(schema.arguments.len(), 2);
        assert!(schema.arguments[0].required);
        assert!(!schema.arguments[1].required);
    }
}
