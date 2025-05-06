//! History Module: Command history management for MCP-ChatBot
//!
//! This module provides functionality for:
//! - Storing and retrieving command history
//! - Persistent storage of history to disk
//! - Thread-safe history access and modification
//! - Navigation through historical commands
//!
//! Key Components:
//! - `History`: Main struct managing command history
//! - File-based persistence using .mcp_history
//! - Thread-safe state via Arc<Mutex>
//!
//! Author: arkSong <arksong2018@gmail.com>
//! Version: 1.0.0
//! License: MIT

use anyhow::Result;
use serde_json::Value;
use std::fs;
use std::sync::Arc;
use tokio::sync::Mutex;

const HISTORY_FILE: &str = "history.txt";

/// Represents a command history with navigation capabilities
pub struct History {
    records: Arc<Mutex<Vec<String>>>,
    current_index: Arc<Mutex<usize>>,
    file_path: String,
}

impl History {
    /// Creates a new History instance
    pub fn new() -> Self {
        let records = Arc::new(Mutex::new(Vec::new()));
        let current_index = Arc::new(Mutex::new(0));
        let file_path = String::new();

        Self {
            records,
            current_index,
            file_path,
        }
    }

    /// Creates a new History instance with a specific file path
    pub fn with_file(file_path: String) -> Self {
        let records = Arc::new(Mutex::new(Vec::new()));
        let current_index = Arc::new(Mutex::new(0));
        let history = Self {
            records,
            current_index,
            file_path,
        };

        // Load existing history if available
        if let Ok(contents) = fs::read_to_string(&history.file_path) {
            let mut records = Vec::new();
            for line in contents.lines() {
                if !line.trim().is_empty() {
                    records.push(line.to_string());
                }
            }
            let mut records_guard = history.records.blocking_lock();
            *records_guard = records;
            let mut index = history.current_index.blocking_lock();
            *index = records_guard.len();
        }

        history
    }

    /// Saves the current history to a file
    pub async fn save(&self) {
        if self.file_path.is_empty() {
            return;
        }
        let records = self.records.lock().await;
        if records.is_empty() {
            return;
        }
        let contents = records.join("\n");
        let _ = fs::write(&self.file_path, format!("{}\n", contents));
    }

    /// Adds a new command to the history
    pub async fn add(&self, input: String) {
        if input.trim().is_empty() {
            return;
        }
        let mut records = self.records.lock().await;
        records.push(input);
        let mut index = self.current_index.lock().await;
        *index = records.len();
        self.save().await;
    }

    /// Gets the previous command in history
    pub async fn get_previous(&self) -> Option<String> {
        let records = self.records.lock().await;
        let mut index = self.current_index.lock().await;

        if records.is_empty() {
            return None;
        }

        if *index >= records.len() {
            *index = records.len() - 1;
        }

        if *index > 0 {
            *index -= 1;
            Some(records[*index].clone())
        } else {
            None
        }
    }

    /// Gets the next command in history
    pub async fn get_next(&self) -> Option<String> {
        let records = self.records.lock().await;
        let mut index = self.current_index.lock().await;

        if records.is_empty() {
            return None;
        }

        if *index < records.len() - 1 {
            *index += 1;
            Some(records[*index].clone())
        } else {
            None
        }
    }

    /// Gets the number of records in history
    pub async fn len(&self) -> usize {
        let records = self.records.lock().await;
        records.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::runtime::Runtime;

    const TEST_HISTORY_FILE: &str = "test_history.txt";

    fn setup() -> (History, Runtime) {
        // Clean up any existing history file
        let _ = fs::remove_file(TEST_HISTORY_FILE);

        let rt = Runtime::new().unwrap();
        let history = History::new();
        (history, rt)
    }

    fn setup_with_file() -> (History, Runtime) {
        // Clean up any existing history file
        let _ = fs::remove_file(TEST_HISTORY_FILE);

        let rt = Runtime::new().unwrap();
        let history = History::with_file(TEST_HISTORY_FILE.to_string());
        (history, rt)
    }

    fn cleanup() {
        let _ = fs::remove_file(TEST_HISTORY_FILE);
    }

    #[test]
    fn test_new_history() {
        let (history, rt) = setup();
        let len = rt.block_on(history.len());
        assert_eq!(len, 0);
        cleanup();
    }

    #[test]
    fn test_add_and_save() {
        let (history, rt) = setup_with_file();

        rt.block_on(async {
            history.add("test command 1".to_string()).await;
            history.add("test command 2".to_string()).await;
        });

        let contents = fs::read_to_string(TEST_HISTORY_FILE).unwrap_or_default();
        assert!(contents.contains("test command 1"));
        assert!(contents.contains("test command 2"));

        cleanup();
    }

    #[test]
    fn test_navigation() {
        let (history, rt) = setup();

        rt.block_on(async {
            history.add("command 1".to_string()).await;
            history.add("command 2".to_string()).await;
            history.add("command 3".to_string()).await;

            let prev1 = history.get_previous().await;
            assert_eq!(prev1, Some("command 2".to_string()));

            let prev2 = history.get_previous().await;
            assert_eq!(prev2, Some("command 1".to_string()));

            let next1 = history.get_next().await;
            assert_eq!(next1, Some("command 2".to_string()));

            let next2 = history.get_next().await;
            assert_eq!(next2, Some("command 3".to_string()));
        });

        cleanup();
    }

    #[test]
    fn test_empty_input() {
        let (history, rt) = setup();

        rt.block_on(async {
            history.add("command 1".to_string()).await;
            history.add("".to_string()).await;
            history.add("   ".to_string()).await;
            history.add("command 2".to_string()).await;

            let len = history.len().await;
            assert_eq!(len, 2);
        });

        cleanup();
    }

    #[test]
    fn test_load_existing_history() {
        cleanup(); // Clean up any existing history file

        // Create a history file with some existing commands
        fs::write(
            TEST_HISTORY_FILE,
            "existing command 1\nexisting command 2\n",
        )
        .unwrap();

        // Create a new history instance that should load the existing file
        let history = History::with_file(TEST_HISTORY_FILE.to_string());
        let rt = Runtime::new().unwrap();

        rt.block_on(async {
            let len = history.len().await;
            assert_eq!(len, 2);

            let prev = history.get_previous().await;
            assert_eq!(prev, Some("existing command 1".to_string()));
        });

        cleanup();
    }
}
