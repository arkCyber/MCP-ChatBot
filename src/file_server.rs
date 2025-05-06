// File server module provides functionality for file system operations
// including reading, writing, deleting files and listing directory contents.
// All operations are performed relative to a root directory for security.
//
// # Usage Example
// ```rust
// use mcp_chatbot::file_server::FileServer;
// use serde_json::json;
// use std::time::Duration;
//
// #[tokio::main]
// async fn main() -> Result<(), Box<dyn std::error::Error>> {
//     // Create a new file server with a root directory
//     let mut file_server = FileServer::new("/path/to/root").await?;
//     file_server.initialize().await?;
//
//     // Write a file
//     let write_args = json!({
//         "path": "test.txt",
//         "content": "Hello, World!"
//     });
//     let write_result = file_server.execute_tool("file_write", &write_args, 3, Duration::from_secs(1)).await?;
//     assert_eq!(write_result["success"], true);
//
//     // Read the file
//     let read_args = json!({
//         "path": "test.txt"
//     });
//     let read_result = file_server.execute_tool("file_read", &read_args, 3, Duration::from_secs(1)).await?;
//     println!("File content: {}", read_result["content"]);
//
//     // Cleanup
//     file_server.cleanup().await?;
//     Ok(())
// }
// ```

use crate::protocol::ResourceSchema;
use anyhow::{Context, Error, Result};
use log::{error, info};
use serde_json::{json, Value};
use std::fs;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::mcp_server::{McpServer, Tool};

/// FileServer struct represents a file system server that provides
/// basic file operations through a set of tools.
/// It uses a root directory to sandbox all operations for security.
///
/// # Available Tools
/// * `file_read` - Read contents of a file
/// * `file_write` - Write contents to a file
/// * `file_delete` - Delete a file
/// * `list_directory` - List contents of a directory
///
/// # Tool Arguments and Responses
/// ## file_read
/// * Arguments: `{ "path": "string" }`
/// * Response: `{ "content": "string", "exists": boolean }`
///
/// ## file_write
/// * Arguments: `{ "path": "string", "content": "string" }`
/// * Response: `{ "success": boolean }`
///
/// ## file_delete
/// * Arguments: `{ "path": "string" }`
/// * Response: `{ "success": boolean, "deleted": boolean }`
///
/// ## list_directory
/// * Arguments: `{ "path": "string" }`
/// * Response: `{ "entries": [{ "name": "string", "type": "file|directory", "size": number, "modified": "string" }] }`
pub struct FileServer {
    /// Base directory for all file operations
    root_path: PathBuf,
    /// MCP server instance that handles tool registration and execution
    mcp_server: McpServer,
}

impl FileServer {
    /// Creates a new FileServer instance with the specified root directory.
    ///
    /// # Arguments
    /// * `root_path` - Base directory path for all file operations
    ///
    /// # Returns
    /// * `Result<Self>` - New FileServer instance or error
    ///
    /// # Example
    /// ```rust
    /// let file_server = FileServer::new("/path/to/root").await?;
    /// ```
    pub async fn new(root_path: &str) -> Result<Self> {
        let root_path = PathBuf::from(root_path);

        // Create root directory if it doesn't exist
        if !root_path.exists() {
            fs::create_dir_all(&root_path)?;
        }

        let mut mcp_server = McpServer::new();
        Self::register_tools(&mut mcp_server, root_path.clone()).await?;

        Ok(Self {
            root_path,
            mcp_server,
        })
    }

    /// Registers all file operation tools with the MCP server.
    /// This includes tools for reading, writing, deleting files and listing directories.
    ///
    /// # Arguments
    /// * `mcp_server` - MCP server instance to register tools with
    /// * `root_path` - Base directory path for all file operations
    ///
    /// # Returns
    /// * `Result<()>` - Success or error during tool registration
    ///
    /// # Tool Registration Details
    /// Each tool is registered with:
    /// * A unique name
    /// * A description
    /// * Input schema (JSON Schema)
    /// * Output schema (JSON Schema)
    /// * An async handler function
    async fn register_tools(mcp_server: &mut McpServer, root_path: PathBuf) -> Result<()> {
        // Clone root_path for each tool's closure to avoid ownership issues
        let read_root_path = root_path.clone();

        // Register read file tool
        // This tool reads the contents of a file and returns them along with existence status
        let read_tool = Tool::new(
            "file_read".to_string(),
            "Read contents of a file".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["path"]
            }),
            Some(json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Contents of the file"
                    },
                    "exists": {
                        "type": "boolean",
                        "description": "Whether the file exists"
                    }
                }
            })),
            Box::new(move |args| {
                let root_path = read_root_path.clone();
                Box::pin(async move {
                    let path = args["path"]
                        .as_str()
                        .ok_or_else(|| Error::msg("Invalid argument: path"))?;

                    let full_path = root_path.join(path);
                    if !full_path.exists() {
                        return Ok(json!({
                            "content": "",
                            "exists": false
                        }));
                    }

                    let content = fs::read_to_string(&full_path)?;
                    Ok(json!({
                        "content": content,
                        "exists": true
                    }))
                })
            }),
        );

        // Clone root_path for write tool
        let write_root_path = root_path.clone();

        // Register write file tool
        // This tool writes content to a file, creating parent directories if needed
        let write_tool = Tool::new(
            "file_write".to_string(),
            "Write contents to a file".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }),
            Some(json!({
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "description": "Whether the write was successful"
                    }
                }
            })),
            Box::new(move |args| {
                let root_path = write_root_path.clone();
                Box::pin(async move {
                    let path = args["path"]
                        .as_str()
                        .ok_or_else(|| Error::msg("Invalid argument: path"))?;
                    let content = args["content"]
                        .as_str()
                        .ok_or_else(|| Error::msg("Invalid argument: content"))?;

                    let full_path = root_path.join(path);

                    // Create parent directories if they don't exist
                    if let Some(parent) = full_path.parent() {
                        fs::create_dir_all(parent)?;
                    }

                    fs::write(&full_path, content)?;
                    Ok(json!({ "success": true }))
                })
            }),
        );

        // Clone root_path for delete tool
        let delete_root_path = root_path.clone();

        // Register delete file tool
        // This tool deletes a file if it exists
        let delete_tool = Tool::new(
            "file_delete".to_string(),
            "Delete a file".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to delete"
                    }
                },
                "required": ["path"]
            }),
            Some(json!({
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "description": "Whether the deletion was successful"
                    },
                    "deleted": {
                        "type": "boolean",
                        "description": "Whether the file was found and deleted"
                    }
                }
            })),
            Box::new(move |args| {
                let root_path = delete_root_path.clone();
                Box::pin(async move {
                    let path = args["path"]
                        .as_str()
                        .ok_or_else(|| Error::msg("Invalid argument: path"))?;

                    let full_path = root_path.join(path);
                    if !full_path.exists() {
                        return Ok(json!({
                            "success": true,
                            "deleted": false
                        }));
                    }

                    fs::remove_file(&full_path)?;
                    Ok(json!({
                        "success": true,
                        "deleted": true
                    }))
                })
            }),
        );

        // Clone root_path for list tool
        let list_root_path = root_path.clone();

        // Register list directory tool
        // This tool lists all entries in a directory with their metadata
        let list_tool = Tool::new(
            "list_directory".to_string(),
            "List contents of a directory".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory to list (relative to root)"
                    }
                },
                "required": ["path"]
            }),
            Some(json!({
                "type": "object",
                "properties": {
                    "entries": {
                        "type": "array",
                        "description": "List of directory entries",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Name of the entry"
                                },
                                "type": {
                                    "type": "string",
                                    "description": "Type of the entry (file/directory)"
                                },
                                "size": {
                                    "type": "integer",
                                    "description": "Size of the entry in bytes (for files)"
                                },
                                "modified": {
                                    "type": "string",
                                    "description": "Last modification time"
                                }
                            }
                        }
                    }
                }
            })),
            Box::new(move |args| {
                let root_path = list_root_path.clone();
                Box::pin(async move {
                    let path = args["path"]
                        .as_str()
                        .ok_or_else(|| Error::msg("Invalid argument: path"))?;

                    let full_path = root_path.join(path);
                    if !full_path.exists() {
                        return Ok(json!({ "entries": [] }));
                    }

                    let entries = fs::read_dir(&full_path)?;
                    let mut result = Vec::new();

                    // Collect metadata for each directory entry
                    for entry in entries {
                        if let Ok(entry) = entry {
                            let name = entry.file_name().to_string_lossy().into_owned();
                            let metadata = entry.metadata()?;
                            let is_dir = metadata.is_dir();
                            let size = if !is_dir { metadata.len() } else { 0 };
                            let modified = metadata.modified()?;
                            let modified_str = format!("{:?}", modified);

                            result.push(json!({
                                "name": name,
                                "type": if is_dir { "directory" } else { "file" },
                                "size": size,
                                "modified": modified_str
                            }));
                        }
                    }

                    Ok(json!({ "entries": result }))
                })
            }),
        );

        // Register all tools with the MCP server
        mcp_server.register_tool(read_tool).await?;
        mcp_server.register_tool(write_tool).await?;
        mcp_server.register_tool(delete_tool).await?;
        mcp_server.register_tool(list_tool).await?;

        Ok(())
    }

    /// Initializes the file server by initializing the underlying MCP server
    ///
    /// # Example
    /// ```rust
    /// let mut file_server = FileServer::new("/path/to/root").await?;
    /// file_server.initialize().await?;
    /// ```
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing file server");
        self.mcp_server.initialize().await
    }

    /// Cleans up resources used by the file server
    ///
    /// # Example
    /// ```rust
    /// file_server.cleanup().await?;
    /// ```
    pub async fn cleanup(&mut self) -> Result<()> {
        info!("Cleaning up file server");
        self.mcp_server.cleanup().await
    }

    /// Executes a tool with the given name and arguments
    ///
    /// # Arguments
    /// * `name` - Name of the tool to execute
    /// * `arguments` - Tool arguments as JSON value
    /// * `retries` - Number of times to retry on failure
    /// * `delay` - Delay between retries
    ///
    /// # Returns
    /// * `Result<Value>` - Tool execution result as JSON value
    ///
    /// # Example
    /// ```rust
    /// // Write a file
    /// let write_args = json!({
    ///     "path": "test.txt",
    ///     "content": "Hello, World!"
    /// });
    /// let write_result = file_server.execute_tool("file_write", &write_args, 3, Duration::from_secs(1)).await?;
    ///
    /// // Read a file
    /// let read_args = json!({
    ///     "path": "test.txt"
    /// });
    /// let read_result = file_server.execute_tool("file_read", &read_args, 3, Duration::from_secs(1)).await?;
    /// ```
    pub async fn execute_tool(
        &mut self,
        name: &str,
        arguments: &Value,
        retries: u32,
        delay: std::time::Duration,
    ) -> Result<Value> {
        self.mcp_server
            .execute_tool(name, arguments, retries, delay)
            .await
    }

    /// Lists all available tools
    ///
    /// # Returns
    /// * `Result<Vec<ToolSchema>>` - List of available tools and their schemas
    ///
    /// # Example
    /// ```rust
    /// let tools = file_server.list_tools().await?;
    /// for tool in tools {
    ///     println!("Tool: {}", tool.name);
    ///     println!("Description: {}", tool.description);
    /// }
    /// ```
    pub async fn list_tools(&self) -> Result<Vec<crate::protocol::ToolSchema>> {
        self.mcp_server.list_tools().await
    }
}

/// Test module for file server functionality
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Tests basic file operations (write, read, list, delete)
    ///
    /// # Test Steps
    /// 1. Create a temporary directory
    /// 2. Initialize file server
    /// 3. Write a test file
    /// 4. Read the file and verify content
    /// 5. List directory and verify entry
    /// 6. Delete the file
    /// 7. Cleanup resources
    #[tokio::test]
    async fn test_file_operations() {
        // Create a temporary directory for testing
        let temp_dir = TempDir::new().unwrap();
        let root_path = temp_dir.path().to_str().unwrap();

        // Initialize file server
        let mut file_server = FileServer::new(root_path).await.unwrap();
        file_server.initialize().await.unwrap();

        // Test write file
        let write_args = json!({
            "path": "test.txt",
            "content": "Hello, World!"
        });
        let write_result = file_server
            .execute_tool(
                "file_write",
                &write_args,
                3,
                std::time::Duration::from_secs(1),
            )
            .await
            .unwrap();
        assert_eq!(write_result["success"], true);

        // Test read file
        let read_args = json!({
            "path": "test.txt"
        });
        let read_result = file_server
            .execute_tool(
                "file_read",
                &read_args,
                3,
                std::time::Duration::from_secs(1),
            )
            .await
            .unwrap();
        assert_eq!(read_result["exists"], true);
        assert_eq!(read_result["content"], "Hello, World!");

        // Test list directory
        let list_args = json!({
            "path": "."
        });
        let list_result = file_server
            .execute_tool(
                "list_directory",
                &list_args,
                3,
                std::time::Duration::from_secs(1),
            )
            .await
            .unwrap();
        let entries = list_result["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["name"], "test.txt");
        assert_eq!(entries[0]["type"], "file");

        // Test delete file
        let delete_args = json!({
            "path": "test.txt"
        });
        let delete_result = file_server
            .execute_tool(
                "file_delete",
                &delete_args,
                3,
                std::time::Duration::from_secs(1),
            )
            .await
            .unwrap();
        assert_eq!(delete_result["success"], true);
        assert_eq!(delete_result["deleted"], true);

        // Cleanup
        file_server.cleanup().await.unwrap();
    }

    /// Tests operations on non-existent files
    ///
    /// # Test Steps
    /// 1. Create a temporary directory
    /// 2. Initialize file server
    /// 3. Attempt to read a non-existent file
    /// 4. Attempt to delete a non-existent file
    /// 5. Cleanup resources
    #[tokio::test]
    async fn test_nonexistent_file() {
        let temp_dir = TempDir::new().unwrap();
        let root_path = temp_dir.path().to_str().unwrap();

        let mut file_server = FileServer::new(root_path).await.unwrap();
        file_server.initialize().await.unwrap();

        // Test reading non-existent file
        let read_args = json!({
            "path": "nonexistent.txt"
        });
        let read_result = file_server
            .execute_tool(
                "file_read",
                &read_args,
                3,
                std::time::Duration::from_secs(1),
            )
            .await
            .unwrap();
        assert_eq!(read_result["exists"], false);
        assert_eq!(read_result["content"], "");

        // Test deleting non-existent file
        let delete_args = json!({
            "path": "nonexistent.txt"
        });
        let delete_result = file_server
            .execute_tool(
                "file_delete",
                &delete_args,
                3,
                std::time::Duration::from_secs(1),
            )
            .await
            .unwrap();
        assert_eq!(delete_result["success"], true);
        assert_eq!(delete_result["deleted"], false);

        file_server.cleanup().await.unwrap();
    }
}
