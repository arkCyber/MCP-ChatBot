//! LLM Client Module: Interface for interacting with Large Language Models
//!
//! This module provides functionality for:
//! - Connecting to multiple LLM APIs (OpenAI, Anthropic, Azure, Deepseek)
//! - Managing API authentication
//! - Sending prompts and receiving responses
//! - Handling request/response formatting
//!
//! Key Components:
//! - `LLMClient`: Main client struct for LLM interactions
//! - API request/response handling
//! - Message formatting and processing
//! - Error handling and logging
//!
//! Author: arkSong <arksong2018@gmail.com>
//! Version: 1.0.0
//! License: MIT

use anyhow::{Error, Result};
use log::{debug, error, info};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

#[derive(Debug, Clone, PartialEq)]
pub enum LLMProvider {
    Ollama,
    OpenAI,
    Deepseek,
}

#[derive(Debug, Deserialize)]
pub struct LLMConfig {
    pub server: ServerConfig,
    pub api_keys: ApiKeys,
    pub models: ModelsConfig,
    pub rate_limits: RateLimits,
    pub logging: LoggingConfig,
    pub cache: CacheConfig,
    pub endpoints: EndpointsConfig,
}

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    pub base_url: String,
    pub timeout: i32,
    pub max_retries: u32,
}

#[derive(Debug, Deserialize)]
pub struct ApiKeys {
    pub openai: String,
    pub anthropic: String,
    pub azure: String,
    pub deepseek: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelsConfig {
    pub default: String,
    pub available: Vec<ModelConfig>,
    pub ollama: String,
    pub openai: String,
    pub deepseek: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub specialized: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RateLimits {
    pub requests_per_minute: u32,
    pub tokens_per_minute: u32,
}

#[derive(Debug, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub file: String,
    pub max_size: u32,
    pub max_backups: u32,
}

#[derive(Debug, Deserialize)]
pub struct CacheConfig {
    pub enabled: bool,
    pub ttl: u32,
    pub max_size: u32,
}

#[derive(Debug, Deserialize)]
pub struct EndpointsConfig {
    pub ollama: String,
    pub openai: String,
    pub deepseek: String,
}

pub struct LLMClient {
    config: LLMConfig,
    provider: LLMProvider,
    client: Client,
}

impl LLMClient {
    pub fn new(config: LLMConfig, provider: LLMProvider) -> Self {
        let timeout = if config.server.timeout < 0 {
            Duration::from_secs(u64::MAX)
        } else {
            Duration::from_secs(config.server.timeout as u64)
        };

        let client = Client::builder()
            .timeout(timeout)
            .no_proxy()
            .build()
            .expect("Failed to create HTTP client");

        info!("Initializing LLM client for provider: {:?}", provider);
        info!("Base URL: {}", config.server.base_url);

        Self {
            config,
            provider,
            client,
        }
    }

    pub fn get_api_key(&self) -> &str {
        match self.provider {
            LLMProvider::OpenAI => &self.config.api_keys.openai,
            LLMProvider::Deepseek => &self.config.api_keys.deepseek,
            LLMProvider::Ollama => "ollama", // Ollama doesn't need an API key
        }
    }

    pub fn get_model_config(&self, model_name: &str) -> Option<&ModelConfig> {
        self.config
            .models
            .available
            .iter()
            .find(|m| m.name == model_name)
    }

    pub async fn get_response(
        &mut self,
        messages: &[Value],
        _model_name: Option<&str>,
    ) -> Result<String> {
        let model = match self.provider {
            LLMProvider::Ollama => &self.config.models.ollama,
            LLMProvider::OpenAI => &self.config.models.openai,
            LLMProvider::Deepseek => &self.config.models.deepseek,
        };

        let model_config = ModelConfig {
            name: model.clone(),
            max_tokens: 4096,
            temperature: 0.7,
            specialized: None,
        };

        let url = match self.provider {
            LLMProvider::OpenAI => self.config.endpoints.openai.clone(),
            LLMProvider::Deepseek => self.config.endpoints.deepseek.clone(),
            LLMProvider::Ollama => self.config.endpoints.ollama.clone(),
        };

        let formatted_messages = self.format_messages(messages)?;
        let request_body = self.create_request_body(model, &model_config, &formatted_messages)?;

        let response = self.send_request(&url, &request_body).await?;
        let response_text = response.text().await?;

        if response_text.is_empty() {
            error!("Received empty response from server");
            return Err(Error::msg("Empty response from server"));
        }

        match self.provider {
            LLMProvider::Ollama => {
                let mut full_response = String::new();
                let mut is_done = false;
                let mut has_content = false;

                for line in response_text.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }

                    if let Ok(response_json) = serde_json::from_str::<Value>(line) {
                        if let Some(message) = response_json.get("message") {
                            if let Some(content) = message.get("content") {
                                if let Some(text) = content.as_str() {
                                    full_response.push_str(text);
                                    has_content = true;
                                }
                            }
                        }

                        if let Some(done) = response_json.get("done") {
                            is_done = done.as_bool().unwrap_or(false);
                            if is_done {
                                break;
                            }
                        }
                    }
                }

                if has_content {
                    Ok(full_response)
                } else {
                    Err(Error::msg("Invalid Ollama response format"))
                }
            }
            LLMProvider::OpenAI | LLMProvider::Deepseek => {
                match serde_json::from_str::<Value>(&response_text) {
                    Ok(response_json) => match self.provider {
                        LLMProvider::OpenAI => response_json["choices"][0]["message"]["content"]
                            .as_str()
                            .map(|s| s.to_string())
                            .ok_or_else(|| Error::msg("Invalid OpenAI response format")),
                        LLMProvider::Deepseek => response_json["choices"][0]["message"]["content"]
                            .as_str()
                            .map(|s| s.to_string())
                            .ok_or_else(|| Error::msg("Invalid Deepseek response format")),
                        _ => unreachable!(),
                    },
                    Err(e) => {
                        error!("Failed to parse response as JSON: {}", e);
                        Err(Error::msg(format!("Failed to parse response: {}", e)))
                    }
                }
            }
        }
    }

    fn format_messages(&self, messages: &[Value]) -> Result<Vec<Value>> {
        let formatted = messages
            .iter()
            .filter(|msg| {
                let content = msg["content"].as_str().unwrap_or("");
                !(msg["role"].as_str().unwrap_or("") == "system" && content.trim().is_empty())
            })
            .map(|msg| {
                let role = msg["role"].as_str().unwrap_or("user");
                let content = msg["content"].as_str().unwrap_or("");

                match self.provider {
                    LLMProvider::Ollama => {
                        if role == "system" {
                            if !content.trim().is_empty() {
                                json!({
                                    "role": "user",
                                    "content": format!("System: {}", content)
                                })
                            } else {
                                json!({
                                    "role": "user",
                                    "content": "Hello"
                                })
                            }
                        } else {
                            json!({
                                "role": role,
                                "content": content
                            })
                        }
                    }
                    _ => json!({
                        "role": role,
                        "content": content
                    }),
                }
            })
            .collect();

        debug!("Formatted messages: {:?}", formatted);
        Ok(formatted)
    }

    fn create_request_body(
        &self,
        model: &str,
        config: &ModelConfig,
        messages: &[Value],
    ) -> Result<Value> {
        let body = match self.provider {
            LLMProvider::Ollama => {
                // info!("Creating Ollama request body with model: {}", model);
                // info!(
                //     "Messages to send: {}",
                //     serde_json::to_string_pretty(messages)?
                // );
                json!({
                    "model": model,
                    "messages": messages,
                    "stream": true
                })
            }
            LLMProvider::OpenAI => json!({
                "model": model,
                "messages": messages,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "stream": false
            }),
            LLMProvider::Deepseek => json!({
                "model": model,
                "messages": messages,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "stream": false
            }),
        };

        Ok(body)
    }

    async fn send_request(&self, url: &str, body: &Value) -> Result<reqwest::Response> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );
        headers.insert(reqwest::header::ACCEPT, "application/json".parse().unwrap());

        match self.provider {
            LLMProvider::OpenAI => {
                headers.insert(
                    reqwest::header::AUTHORIZATION,
                    format!("Bearer {}", self.get_api_key()).parse().unwrap(),
                );
            }
            LLMProvider::Deepseek => {
                headers.insert(
                    reqwest::header::AUTHORIZATION,
                    format!("Bearer {}", self.get_api_key()).parse().unwrap(),
                );
            }
            LLMProvider::Ollama => {}
        }

        let response = self
            .client
            .post(url)
            .headers(headers)
            .json(body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            error!("Error response from server: {}", error_text);
            return Err(Error::msg(format!("Server returned error: {}", error_text)));
        }

        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> LLMConfig {
        LLMConfig {
            server: ServerConfig {
                base_url: "https://api.test.com".to_string(),
                timeout: 30,
                max_retries: 3,
            },
            api_keys: ApiKeys {
                openai: "test_openai_key".to_string(),
                anthropic: "test_anthropic_key".to_string(),
                azure: "test_azure_key".to_string(),
                deepseek: "test_deepseek_key".to_string(),
            },
            models: ModelsConfig {
                default: "gpt-4".to_string(),
                available: vec![ModelConfig {
                    name: "gpt-4".to_string(),
                    max_tokens: 8192,
                    temperature: 0.7,
                    specialized: None,
                }],
                ollama: "llama3.2:latest".to_string(),
                openai: "https://api.test.com/v1/chat/completions".to_string(),
                deepseek: "https://api.test.com/v1/chat/completions".to_string(),
            },
            rate_limits: RateLimits {
                requests_per_minute: 60,
                tokens_per_minute: 90000,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                file: "test.log".to_string(),
                max_size: 100,
                max_backups: 5,
            },
            cache: CacheConfig {
                enabled: true,
                ttl: 3600,
                max_size: 1000,
            },
            endpoints: EndpointsConfig {
                ollama: "http://localhost:11434/api/chat".to_string(),
                openai: "https://api.test.com/v1/chat/completions".to_string(),
                deepseek: "https://api.test.com/v1/chat/completions".to_string(),
            },
        }
    }

    #[test]
    fn test_get_api_key() {
        let config = create_test_config();
        let client = LLMClient::new(config, LLMProvider::OpenAI);
        assert_eq!(client.get_api_key(), "test_openai_key");
    }

    #[test]
    fn test_get_model_config() {
        let config = create_test_config();
        let client = LLMClient::new(config, LLMProvider::OpenAI);

        let model_config = client.get_model_config("gpt-4").unwrap();
        assert_eq!(model_config.name, "gpt-4");
        assert_eq!(model_config.max_tokens, 8192);
        assert_eq!(model_config.temperature, 0.7);
    }
}
