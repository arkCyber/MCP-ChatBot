use crate::llm_client::LLMConfig;
use anyhow::Result;
use log::{debug, warn};
use serde::{Deserialize, Serialize};
use serde_json;
use serde_json::Value;
use serde_yaml;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct Configuration {
    pub api_key: Option<String>, // Optional API key for LLM services
}

impl Configuration {
    pub fn new() -> Self {
        Self { api_key: None }
    }

    pub fn load_config(&self, config_path: &str) -> anyhow::Result<serde_json::Value> {
        let config_str = std::fs::read_to_string(config_path)?;
        let config: serde_json::Value = serde_json::from_str(&config_str)?;
        Ok(config)
    }

    pub fn load_llm_config(
        &self,
        config_path: &str,
    ) -> anyhow::Result<crate::llm_client::LLMConfig> {
        let config_str = std::fs::read_to_string(config_path)?;
        let config: crate::llm_client::LLMConfig = serde_yaml::from_str(&config_str)?;

        // Validate configuration
        if config.server.base_url.is_empty() {
            warn!("Server base URL is empty");
        }

        if config.models.default.is_empty() {
            warn!("Default model is not specified");
        }

        if config.models.available.is_empty() {
            warn!("No available models configured");
        }

        debug!("Loaded LLM configuration: {:?}", config);
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_config_file() -> NamedTempFile {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"
server:
  base_url: "https://api.test.com"
  timeout: 30
  max_retries: 3

api_keys:
  openai: "test_openai_key"
  anthropic: "test_anthropic_key"
  azure: "test_azure_key"
  deepseek: "test_deepseek_key"

models:
  default: "gpt-4"
  available:
    - name: "gpt-4"
      max_tokens: 8192
      temperature: 0.7
    - name: "gpt-3.5-turbo"
      max_tokens: 4096
      temperature: 0.7

rate_limits:
  requests_per_minute: 60
  tokens_per_minute: 90000

logging:
  level: "info"
  file: "test.log"
  max_size: 100
  max_backups: 5

cache:
  enabled: true
  ttl: 3600
  max_size: 1000
"#
        )
        .unwrap();
        temp_file
    }

    #[test]
    fn test_load_valid_config() {
        let temp_file = create_test_config_file();
        let config = Configuration::new();
        let llm_config = config
            .load_llm_config(temp_file.path().to_str().unwrap())
            .unwrap();

        assert_eq!(llm_config.server.base_url, "https://api.test.com");
        assert_eq!(llm_config.server.timeout, 30);
        assert_eq!(llm_config.api_keys.openai, "test_openai_key");
        assert_eq!(llm_config.models.default, "gpt-4");
        assert_eq!(llm_config.models.available.len(), 2);
    }

    #[test]
    fn test_configuration_new() {
        let config = Configuration::new();
        assert!(config.api_key.is_none());
    }
}
