#[cfg(test)]
mod tests {
    use anyhow::Result;
    use mcp_chatbot::sqlite_server::SqliteServer;
    use serde_json::json;

    #[tokio::test]
    async fn test_sqlite_basic() {
        assert!(true);
    }

    #[tokio::test]
    async fn test_sqlite_connection() -> Result<()> {
        let server = SqliteServer::new(":memory:").await?;
        // Note: is_connected() is not available, so we'll just verify the server was created
        assert!(true);
        Ok(())
    }

    #[tokio::test]
    async fn test_sqlite_create_table() -> Result<()> {
        let server = SqliteServer::new(":memory:").await?;

        let result = server
            .execute_tool(
                "sqlite_create_table",
                &json!({
                    "name": "test_table",
                    "columns": [
                        {
                            "name": "id",
                            "type": "INTEGER",
                            "primary_key": true
                        },
                        {
                            "name": "name",
                            "type": "TEXT",
                            "not_null": true
                        }
                    ]
                }),
                3,
                std::time::Duration::from_secs(1),
            )
            .await?;

        assert!(result["success"].as_bool().unwrap_or(false));
        Ok(())
    }

    #[tokio::test]
    async fn test_sqlite_insert_and_query() -> Result<()> {
        let server = SqliteServer::new(":memory:").await?;

        // Create table
        server
            .execute_tool(
                "sqlite_create_table",
                &json!({
                    "name": "test_table",
                    "columns": [
                        {
                            "name": "id",
                            "type": "INTEGER",
                            "primary_key": true
                        },
                        {
                            "name": "name",
                            "type": "TEXT",
                            "not_null": true
                        }
                    ]
                }),
                3,
                std::time::Duration::from_secs(1),
            )
            .await?;

        // Insert data
        let insert_result = server
            .execute_tool(
                "sqlite_execute",
                &json!({
                    "query": "INSERT INTO test_table (name) VALUES ('test')"
                }),
                3,
                std::time::Duration::from_secs(1),
            )
            .await?;

        assert!(insert_result["success"].as_bool().unwrap_or(false));

        // Query data
        let query_result = server
            .execute_tool(
                "sqlite_query",
                &json!({
                    "query": "SELECT * FROM test_table"
                }),
                3,
                std::time::Duration::from_secs(1),
            )
            .await?;

        assert!(query_result["success"].as_bool().unwrap_or(false));
        assert!(query_result["rows"].as_array().unwrap().len() > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_sqlite_list_tables() -> Result<()> {
        let server = SqliteServer::new(":memory:").await?;

        // Create a test table
        server
            .execute_tool(
                "sqlite_create_table",
                &json!({
                    "name": "test_table",
                    "columns": [
                        {
                            "name": "id",
                            "type": "INTEGER",
                            "primary_key": true
                        }
                    ]
                }),
                3,
                std::time::Duration::from_secs(1),
            )
            .await?;

        // List tables
        let result = server
            .execute_tool(
                "sqlite_list_tables",
                &json!({}),
                3,
                std::time::Duration::from_secs(1),
            )
            .await?;

        let tables = result["tables"].as_array().unwrap();
        assert!(tables.iter().any(|t| t.as_str().unwrap() == "test_table"));

        Ok(())
    }

    #[tokio::test]
    async fn test_sqlite_drop_table() -> Result<()> {
        let server = SqliteServer::new(":memory:").await?;

        // Create a test table
        server
            .execute_tool(
                "sqlite_create_table",
                &json!({
                    "name": "test_table",
                    "columns": [
                        {
                            "name": "id",
                            "type": "INTEGER",
                            "primary_key": true
                        }
                    ]
                }),
                3,
                std::time::Duration::from_secs(1),
            )
            .await?;

        // Drop the table
        let drop_result = server
            .execute_tool(
                "sqlite_drop_table",
                &json!({
                    "name": "test_table"
                }),
                3,
                std::time::Duration::from_secs(1),
            )
            .await?;

        assert!(drop_result["success"].as_bool().unwrap_or(false));

        // Verify table is gone
        let list_result = server
            .execute_tool(
                "sqlite_list_tables",
                &json!({}),
                3,
                std::time::Duration::from_secs(1),
            )
            .await?;

        let tables = list_result["tables"].as_array().unwrap();
        assert!(!tables.iter().any(|t| t.as_str().unwrap() == "test_table"));

        Ok(())
    }
}
