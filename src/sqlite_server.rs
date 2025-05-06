//! SQLite Server Module: Database integration for the MCP-ChatBot framework
//!
//! This module provides SQLite database functionality including:
//! - Key-value store operations
//! - Database connection management
//! - Tool registration for database operations
//!
//! The SQLite server supports:
//! - Creating/opening databases with proper flags
//! - Schema initialization
//! - Thread-safe connection handling
//! - Integration with the MCP server framework
//!
//! Key Components:
//! - `SqliteServer`: Main server struct managing the database connection
//! - Database tools for key-value operations
//! - Connection pooling via Arc<Mutex>
//!
//! Author: arkSong <arksong2018@gmail.com>
//! Version: 1.0.0
//! License: MIT

use anyhow::{Error, Result};
use log::info;
use rusqlite::{Connection, OpenFlags};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

use crate::mcp_server::{McpServer, Tool};

pub struct SqliteServer {
    conn: Arc<Mutex<Connection>>,
    mcp_server: McpServer,
}

impl SqliteServer {
    pub async fn new(db_path: &str) -> Result<Self> {
        // Create database directory if it doesn't exist
        if let Some(parent) = Path::new(db_path).parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let db_path = db_path.to_string();

        // Open SQLite connection with proper flags
        let conn = tokio::task::spawn_blocking(move || {
            let mut conn = Connection::open_with_flags(
                &db_path,
                OpenFlags::SQLITE_OPEN_READ_WRITE
                    | OpenFlags::SQLITE_OPEN_CREATE
                    | OpenFlags::SQLITE_OPEN_URI,
            )?;

            // Initialize database schema
            let tx = conn.transaction()?;
            tx.execute_batch(
                "PRAGMA foreign_keys = ON;
                PRAGMA busy_timeout = 5000;
                CREATE TABLE IF NOT EXISTS kv_store (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );",
            )?;
            tx.commit()?;

            Ok::<_, Error>(conn)
        })
        .await??;

        let conn = Arc::new(Mutex::new(conn));
        let mut mcp_server = McpServer::new();

        // Register SQLite tools
        Self::register_tools(&mut mcp_server, Arc::clone(&conn)).await?;

        Ok(Self { conn, mcp_server })
    }

    async fn register_tools(
        mcp_server: &mut McpServer,
        conn: Arc<Mutex<Connection>>,
    ) -> Result<()> {
        // Register set tool
        let conn_set = Arc::clone(&conn);
        let set_tool = Tool::new(
            "sqlite_set".to_string(),
            "Set a key-value pair in SQLite database".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Key to set"
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to set"
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
            Box::new(move |args| {
                let conn_set = Arc::clone(&conn_set);
                let key = args["key"].as_str().unwrap_or_default().to_string();
                let value = args["value"].as_str().unwrap_or_default().to_string();

                Box::pin(async move {
                    let result = tokio::task::spawn_blocking(move || {
                        let mut conn = conn_set.lock().unwrap();
                        let tx = conn.transaction()?;

                        let result = tx.execute(
                            "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?1, ?2)",
                            [&key, &value],
                        )?;

                        tx.commit()?;
                        Ok::<_, Error>(result)
                    })
                    .await??;

                    Ok(json!({ "success": result > 0 }))
                })
            }),
        );

        // Register get tool
        let conn_get = Arc::clone(&conn);
        let get_tool = Tool::new(
            "sqlite_get".to_string(),
            "Get a value from SQLite database by key".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Key to get"
                    }
                },
                "required": ["key"]
            }),
            Some(json!({
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "The value associated with the key"
                    },
                    "found": {
                        "type": "boolean",
                        "description": "Whether the key was found"
                    }
                }
            })),
            Box::new(move |args| {
                let conn_get = Arc::clone(&conn_get);
                let key = args["key"].as_str().unwrap_or_default().to_string();

                Box::pin(async move {
                    let result = tokio::task::spawn_blocking(move || {
                        let mut conn = conn_get.lock().unwrap();
                        let tx = conn.transaction()?;
                        let mut value = String::new();
                        let mut found = false;

                        {
                            let mut stmt =
                                tx.prepare("SELECT value FROM kv_store WHERE key = ?1")?;
                            let mut rows = stmt.query([&key])?;
                            if let Some(row) = rows.next()? {
                                value = row.get(0)?;
                                found = true;
                            }
                        }

                        tx.commit()?;
                        Ok::<_, Error>((value, found))
                    })
                    .await??;

                    Ok(json!({
                        "value": result.0,
                        "found": result.1
                    }))
                })
            }),
        );

        // Register delete tool
        let conn_delete = Arc::clone(&conn);
        let delete_tool = Tool::new(
            "sqlite_delete".to_string(),
            "Delete a key-value pair from SQLite database".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Key to delete"
                    }
                },
                "required": ["key"]
            }),
            Some(json!({
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "description": "Whether the operation was successful"
                    },
                    "deleted": {
                        "type": "boolean",
                        "description": "Whether the key was found and deleted"
                    }
                }
            })),
            Box::new(move |args| {
                let conn_delete = Arc::clone(&conn_delete);
                let key = args["key"].as_str().unwrap_or_default().to_string();

                Box::pin(async move {
                    let deleted = tokio::task::spawn_blocking(move || {
                        let mut conn = conn_delete.lock().unwrap();
                        let tx = conn.transaction()?;

                        let deleted = tx.execute("DELETE FROM kv_store WHERE key = ?1", [&key])?;

                        tx.commit()?;
                        Ok::<_, Error>(deleted)
                    })
                    .await??;

                    Ok(json!({
                        "success": true,
                        "deleted": deleted > 0
                    }))
                })
            }),
        );

        mcp_server.register_tool(set_tool).await?;
        mcp_server.register_tool(get_tool).await?;
        mcp_server.register_tool(delete_tool).await?;

        Ok(())
    }

    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing SQLite server");
        self.mcp_server.initialize().await
    }

    pub async fn cleanup(&mut self) -> Result<()> {
        info!("Cleaning up SQLite server");
        self.mcp_server.cleanup().await
    }

    pub async fn execute_tool(
        &self,
        tool_name: &str,
        arguments: &Value,
        _retry_count: u32,
        _timeout: std::time::Duration,
    ) -> Result<Value> {
        match tool_name {
            "sqlite_set" => {
                let key = arguments["key"]
                    .as_str()
                    .ok_or_else(|| Error::msg("Invalid argument: key"))?;
                let value = arguments["value"]
                    .as_str()
                    .ok_or_else(|| Error::msg("Invalid argument: value"))?;

                let success = self.set(key, value).await?;
                Ok(json!({ "success": success }))
            }
            "sqlite_get" => {
                let key = arguments["key"]
                    .as_str()
                    .ok_or_else(|| Error::msg("Invalid argument: key"))?;

                let (value, found) = self.get(key).await?;
                Ok(json!({
                    "value": value,
                    "found": found
                }))
            }
            "sqlite_delete" => {
                let key = arguments["key"]
                    .as_str()
                    .ok_or_else(|| Error::msg("Invalid argument: key"))?;

                let deleted = self.delete(key).await?;
                Ok(json!({
                    "success": true,
                    "deleted": deleted
                }))
            }
            _ => Err(Error::msg(format!("Unknown tool: {}", tool_name))),
        }
    }

    pub async fn list_tools(&self) -> Result<Vec<crate::protocol::ToolSchema>> {
        self.mcp_server.list_tools().await
    }

    pub async fn set(&self, key: &str, value: &str) -> Result<bool> {
        let conn = self.conn.clone();
        let key = key.to_string();
        let value = value.to_string();

        let result = tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().unwrap();
            let tx = conn.transaction()?;

            let result = tx.execute(
                "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?1, ?2)",
                [&key, &value],
            )?;

            tx.commit()?;
            Ok::<_, Error>(result)
        })
        .await??;

        Ok(result > 0)
    }

    pub async fn get(&self, key: &str) -> Result<(String, bool)> {
        let conn = self.conn.clone();
        let key = key.to_string();

        let result = tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().unwrap();
            let tx = conn.transaction()?;
            let mut value = String::new();
            let mut found = false;

            {
                let mut stmt = tx.prepare("SELECT value FROM kv_store WHERE key = ?1")?;
                let mut rows = stmt.query([&key])?;
                if let Some(row) = rows.next()? {
                    value = row.get(0)?;
                    found = true;
                }
            }

            tx.commit()?;
            Ok::<_, Error>((value, found))
        })
        .await??;

        Ok(result)
    }

    pub async fn delete(&self, key: &str) -> Result<bool> {
        let conn = self.conn.clone();
        let key = key.to_string();

        let changes = tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().unwrap();
            let tx = conn.transaction()?;

            let changes = tx.execute("DELETE FROM kv_store WHERE key = ?1", [&key])?;

            tx.commit()?;
            Ok::<_, Error>(changes)
        })
        .await??;

        Ok(changes > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::tempdir;

    async fn setup_test_db() -> (SqliteServer, tempfile::TempDir) {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let mut server = SqliteServer::new(db_path.to_str().unwrap()).await.unwrap();
        server.initialize().await.unwrap();
        (server, temp_dir)
    }

    #[tokio::test]
    async fn test_sqlite_server_creation() {
        let (mut server, _temp_dir) = setup_test_db().await;
        assert!(server.initialize().await.is_ok());
    }

    #[tokio::test]
    async fn test_set_get_operations() {
        let (server, _temp_dir) = setup_test_db().await;

        // Test set operation
        let result = server.set("test_key", "test_value").await;
        assert!(result.is_ok());
        assert!(result.unwrap());

        // Test get operation
        let result = server.get("test_key").await;
        assert!(result.is_ok());
        let (value, found) = result.unwrap();
        assert!(found);
        assert_eq!(value, "test_value");

        // Test get non-existent key
        let result = server.get("non_existent_key").await;
        assert!(result.is_ok());
        let (value, found) = result.unwrap();
        assert!(!found);
        assert_eq!(value, "");
    }

    #[tokio::test]
    async fn test_delete_operation() {
        let (server, _temp_dir) = setup_test_db().await;

        // Set a key-value pair
        assert!(server.set("test_key", "test_value").await.unwrap());

        // Delete the key
        let result = server.delete("test_key").await;
        assert!(result.is_ok());
        assert!(result.unwrap());

        // Verify the key is deleted
        let (_, found) = server.get("test_key").await.unwrap();
        assert!(!found);

        // Delete non-existent key
        let result = server.delete("non_existent_key").await;
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[tokio::test]
    async fn test_cleanup() {
        let (mut server, _temp_dir) = setup_test_db().await;
        assert!(server.cleanup().await.is_ok());
    }

    #[tokio::test]
    async fn test_list_tools() {
        let (server, _temp_dir) = setup_test_db().await;
        let tools = server.list_tools().await.unwrap();
        assert_eq!(tools.len(), 3); // sqlite_set, sqlite_get, sqlite_delete
    }

    #[tokio::test]
    async fn test_tool_execution() {
        let (server, _temp_dir) = setup_test_db().await;
        let timeout = Duration::from_secs(1);

        // Test set tool
        let set_args = json!({
            "key": "test_key",
            "value": "test_value"
        });
        let result = server
            .execute_tool("sqlite_set", &set_args, 1, timeout)
            .await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response["success"].as_bool().unwrap());

        // Test get tool
        let get_args = json!({
            "key": "test_key"
        });
        let result = server
            .execute_tool("sqlite_get", &get_args, 1, timeout)
            .await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response["found"].as_bool().unwrap());
        assert_eq!(response["value"].as_str().unwrap(), "test_value");

        // Test delete tool
        let delete_args = json!({
            "key": "test_key"
        });
        let result = server
            .execute_tool("sqlite_delete", &delete_args, 1, timeout)
            .await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response["success"].as_bool().unwrap());
        assert!(response["deleted"].as_bool().unwrap());
    }
}
