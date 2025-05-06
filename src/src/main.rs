impl Server {
    /// Creates a new Server instance
    /// Loads configuration from servers_config.json
    fn new(name: &str) -> Result<Self> {
        let config = fs::read_to_string("../config/servers_config.json")?;
        let config: Value = serde_json::from_str(&config)?;
        Ok(Self {
            name: name.to_string(),
            config,
        })
    }

    // ... existing code ...
}

impl ChatSession {
    /// Creates a new ChatSession instance
    /// Initializes the LLM client and loads system prompts
    pub async fn new(llm_client: Arc<Mutex<LLMClient>>) -> Result<Self> {
        let prompts = load_system_prompts();
        let history = History::new();
        let mut readline = Editor::<(), FileHistory>::new()?;
        readline.set_max_history_size(1000)?;
        if let Err(e) = readline.load_history(".mcp_history") {
            info!("No history file found: {}", e);
        }
        Ok(Self {
            llm_client,
            prompts,
            history,
            readline,
            current_ai_server: "ollama".to_string(),
            running: true,
            cached_tools: Vec::new(),
            servers: Vec::new(),
        })
    }

    // ... existing code ...
}
