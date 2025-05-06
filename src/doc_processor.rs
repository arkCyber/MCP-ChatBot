use anyhow::Result;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Deserialize)]
pub struct ObsidianConfig {
    pub vault_path: String,
    pub file_extensions: Vec<String>,
    pub exclude_patterns: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: serde_json::Value,
    pub path: String,
}

pub struct DocProcessor {
    config: ObsidianConfig,
}

impl DocProcessor {
    pub fn new(config: ObsidianConfig) -> Self {
        Self { config }
    }

    pub fn process_documents(&self) -> Result<Vec<Document>> {
        let mut documents = Vec::new();
        let vault_path = shellexpand::tilde(&self.config.vault_path).to_string();

        for entry in WalkDir::new(&vault_path).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();

            // 检查文件扩展名
            if !self.is_valid_extension(path) {
                continue;
            }

            // 检查排除模式
            if self.should_exclude(path) {
                continue;
            }

            match self.process_file(path) {
                Ok(doc) => documents.push(doc),
                Err(e) => warn!("处理文件失败 {}: {}", path.display(), e),
            }
        }

        info!("处理了 {} 个文档", documents.len());
        Ok(documents)
    }

    fn is_valid_extension(&self, path: &Path) -> bool {
        if let Some(ext) = path.extension() {
            if let Some(ext_str) = ext.to_str() {
                return self
                    .config
                    .file_extensions
                    .contains(&format!(".{}", ext_str));
            }
        }
        false
    }

    fn should_exclude(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        self.config
            .exclude_patterns
            .iter()
            .any(|pattern| path_str.contains(pattern))
    }

    fn process_file(&self, path: &Path) -> Result<Document> {
        let content = fs::read_to_string(path)?;
        let id = self.generate_doc_id(path);
        let metadata = self.extract_metadata(&content);

        Ok(Document {
            id,
            content,
            metadata,
            path: path.to_string_lossy().to_string(),
        })
    }

    fn generate_doc_id(&self, path: &Path) -> String {
        // 使用文件路径的哈希值作为文档ID
        let path_str = path.to_string_lossy();
        format!("{:x}", md5::compute(path_str.as_bytes()))
    }

    fn extract_metadata(&self, content: &str) -> serde_json::Value {
        // 提取文档的元数据（标题、标签等）
        let mut metadata = serde_json::json!({});

        // 提取标题（第一行）
        if let Some(first_line) = content.lines().next() {
            if first_line.starts_with("# ") {
                metadata["title"] = serde_json::Value::String(first_line[2..].to_string());
            }
        }

        // 提取标签
        let tags: Vec<String> = content
            .lines()
            .filter_map(|line| {
                if line.contains("#") {
                    line.split_whitespace()
                        .filter(|word| word.starts_with("#"))
                        .map(|tag| tag.to_string())
                        .collect::<Vec<String>>()
                        .first()
                        .cloned()
                } else {
                    None
                }
            })
            .collect();

        if !tags.is_empty() {
            metadata["tags"] =
                serde_json::Value::Array(tags.into_iter().map(serde_json::Value::String).collect());
        }

        metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_doc_processor() {
        let temp_dir = tempdir().unwrap();
        let test_file = temp_dir.path().join("test.md");
        fs::write(
            &test_file,
            "# Test Document\n\nThis is a test document with #test #tag",
        )
        .unwrap();

        let config = ObsidianConfig {
            vault_path: temp_dir.path().to_string_lossy().to_string(),
            file_extensions: vec![".md".to_string()],
            exclude_patterns: vec![],
        };

        let processor = DocProcessor::new(config);
        let docs = processor.process_documents().unwrap();

        assert_eq!(docs.len(), 1);
        let doc = &docs[0];
        assert_eq!(doc.metadata["title"], "Test Document");
        assert!(doc.metadata["tags"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("#test")));
    }
}
