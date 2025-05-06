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
api_keys:
  openai: "test_openai_key"
  deepseek: "test_deepseek_key"

models:
  ollama: "llama3.2"
  openai: "gpt-3.5-turbo"
  deepseek: "deepseek-chat"

endpoints:
  ollama: "http://localhost:11434/api/chat"
  openai: "https://api.openai.com/v1/chat/completions"
  deepseek: "https://api.deepseek.com/v1/chat/completions"
"#
        )
        .unwrap();
        temp_file
    }

    #[test]
    fn test_load_valid_config() {
        let temp_file = create_test_config_file();
        let config = load_config(temp_file.path()).unwrap();

        assert_eq!(config.api_keys.openai, "test_openai_key");
        assert_eq!(config.api_keys.deepseek, "test_deepseek_key");
        assert_eq!(config.models.ollama, "llama3.2");
        assert_eq!(config.models.openai, "gpt-3.5-turbo");
        assert_eq!(config.models.deepseek, "deepseek-chat");
        assert_eq!(config.endpoints.ollama, "http://localhost:11434/api/chat");
        assert_eq!(
            config.endpoints.openai,
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            config.endpoints.deepseek,
            "https://api.deepseek.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_load_invalid_config() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"
api_keys:
  openai: ""
  deepseek: ""

models:
  ollama: ""
  openai: ""
  deepseek: ""

endpoints:
  ollama: ""
  openai: ""
  deepseek: ""
"#
        )
        .unwrap();

        let result = load_config(temp_file.path());
        assert!(result.is_ok()); // Should still be ok, just with warnings
    }
}
