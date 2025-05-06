# MCP-ChatBot

A powerful Rust-based chatbot MCP (Model Context Protocol) framework with multi-server support and tool integration capabilities.

[![Rust](https://img.shields.io/badge/Rust-1.70+-blue.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/arksong/mcp-chatbot/actions/workflows/rust.yml/badge.svg)](https://github.com/arksong/mcp-chatbot/actions/workflows/rust.yml)

## Features

- 🤖 **Multi-AI Support**: Seamlessly switch between Ollama (local) and OpenAI
- 🛠️ **Tool Integration**: Built-in support for memory, SQLite, and file operations
- 🔄 **Multi-Server Architecture**: Run multiple specialized servers simultaneously
- 💬 **Interactive CLI**: User-friendly command-line interface with history
- 📝 **Customizable Prompts**: Server-specific system prompts via YAML configuration
- 🔒 **Secure**: Environment-based API key management
- 📚 **RAG Support**: Retrieval Augmented Generation with Qdrant vector database
- 🎤 **Voice Input**: Speech-to-text capabilities using Whisper
- 🤖 **Advanced NLP**: Powered by rust-bert for text embeddings and language models
- 🧠 **Deep Learning**: PyTorch integration via tch for advanced model operations
- 📝 **Text Processing**: Efficient tokenization with Hugging Face tokenizers

## Model Context Protocol (MCP)

MCP(Model Context Protocol) is a flexible protocol designed to enhance AI model interactions by providing structured context and tool integration capabilities. The protocol enables:

### Core Components

1. **Context Management**
   - Dynamic context switching between different AI models
   - Context persistence across sessions
   - Server-specific context configurations

2. **Tool Integration**
   - Standardized tool interface for AI models
   - Automatic tool discovery and registration
   - Tool execution with retry mechanisms
   - Tool response processing and formatting

3. **Server Architecture**
   - Modular server design for specialized operations
   - Inter-server communication protocol
   - Resource management and cleanup
   - Server-specific prompt configurations

4. **Protocol Features**
   - JSON-based message format
   - Asynchronous operation support
   - Error handling and recovery
   - Resource cleanup and management
   - Tool execution monitoring

### Protocol Flow

1. **Initialization**
   - Server registration and configuration
   - Tool discovery and registration
   - Context initialization

2. **Operation**
   - Context-aware tool execution
   - Response processing and formatting
   - Error handling and recovery
   - Resource management

3. **Cleanup**
   - Resource release
   - Server shutdown
   - Context persistence

### Use Cases

- **Multi-Model Collaboration**: Coordinate multiple AI models for complex tasks
- **Tool Integration**: Seamlessly integrate external tools and services
- **Context Management**: Maintain consistent context across different operations
- **Resource Management**: Efficiently manage system resources and cleanup

## Prerequisites

- Rust 1.70 or higher
- Ollama (for local AI support)
- OpenAI API key (optional, for OpenAI support)
- Qdrant vector database (for RAG support)
- Whisper model (for voice input)

## AI Models and Tools

The project leverages several powerful AI and NLP tools:

### Text Processing and Embeddings
- **rust-bert**: A Rust implementation of Hugging Face's transformers library
  - Provides state-of-the-art text embeddings
  - Supports multiple language models
  - Enables efficient text processing and understanding

### Deep Learning
- **tch (PyTorch)**: Rust bindings for PyTorch
  - Enables deep learning model operations
  - Supports model inference and training
  - Provides GPU acceleration when available

### Text Tokenization
- **tokenizers**: Hugging Face's tokenizers library
  - Efficient text tokenization
  - Supports multiple tokenization algorithms
  - Enables consistent text processing across different models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/arksong/mcp-chatbot.git
cd mcp-chatbot
```

2. Build the project:
```bash
cargo build --release
```

## Configuration

1. Create a `.env` file in the project root:
```bash
LLM_API_KEY=your_ollama_key
OPENAI_API_KEY=your_openai_key  # Optional
```

2. Configure servers in `src/servers_config.json`

3. Customize prompts in `mcp_prompts.yaml`

## Usage

### Using Ollama (Local AI)

1. Install Ollama:
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

2. Pull the required model:
```bash
ollama pull llama3.2:latest
```

3. Start the Ollama service:
```bash
ollama serve
```

4. Run the chatbot:
```bash
cargo run
```

### Using OpenAI

1. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key
```

2. Run the chatbot:
```bash
cargo run
```

3. Switch to OpenAI using the `/ai` command

### Using RAG (Retrieval Augmented Generation)

1. Start Qdrant vector database using Docker:
```bash
# Using the provided setup script
./scripts/setup_qdrant.sh

# Or manually using Docker
docker run -d \
    --name qdrant \
    -p 6333:6333 \
    -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage" \
    qdrant/qdrant:latest
```

2. Add documents to the RAG database:
```bash
# Use the /rag-add command in the chatbot
/rag-add
# Then enter your document text
```

3. Search similar documents:
```bash
# Use the /rag-search command
/rag-search
# Enter your search query
```

4. View RAG database information:
```bash
/rag-info
```

5. Docker Management Commands:
```bash
# Stop Qdrant container
docker stop qdrant

# Start Qdrant container
docker start qdrant

# View Qdrant logs
docker logs qdrant

# Remove Qdrant container (data will be preserved in qdrant_storage)
docker rm qdrant
```

Note: The Qdrant data is persisted in the `./qdrant_storage` directory, which is mounted as a volume in the Docker container. This ensures your vector data remains intact even if the container is removed.

### Using Voice Input

1. Start voice recording:
```bash
/voice
```

2. Speak your message (press Enter to stop recording)

3. The transcribed text will be processed as a normal message

## Available Commands

- `/help` - Display help menu
- `/clear` - Clear the terminal screen
- `/usage` - Display usage information
- `/exit` - Exit the program
- `/servers` - List available MCP servers
- `/tools` - List available tools
- `/resources` - List available resources
- `/debug` - Toggle debug logging
- `/ai` - Switch between AI providers
- `/rag-add` - Add a new document to RAG database
- `/rag-search` - Search for similar documents
- `/rag-info` - Show RAG database information
- `/voice` - Start voice input (press Enter to stop recording)

## Tool Examples

### Memory Operations
```json
{"tool": "memory_set", "arguments": {"key": "name", "value": "John"}}
{"tool": "memory_get", "arguments": {"key": "name"}}
```

### SQLite Operations
```json
{"tool": "sqlite_create_table", "arguments": {"name": "users", "columns": [{"name": "id", "type": "INTEGER"}]}}
{"tool": "sqlite_query", "arguments": {"query": "SELECT * FROM users"}}
```

### File Operations
```json
{"tool": "file_write", "arguments": {"path": "test.txt", "content": "Hello"}}
{"tool": "file_read", "arguments": {"path": "test.txt"}}
```

## Project Structure

```
mcp-chatbot/
├── src/
│   ├── main.rs           # Main application entry
│   ├── llm_client.rs     # LLM client implementation
│   ├── mcp_server.rs     # MCP server core
│   ├── protocol.rs       # Protocol definitions
│   ├── sqlite_server.rs  # SQLite server implementation
│   ├── stdio_server.rs   # Standard I/O server
│   ├── rag_server.rs     # RAG server implementation
│   ├── whisper_server.rs # Whisper server implementation
│   └── utils.rs          # Utility functions
├── tests/
│   ├── sqlite_test.rs    # SQLite tests
│   └── rag_server_test.rs # RAG server tests
├── Cargo.toml            # Project dependencies
├── mcp_prompts.yaml      # System prompts configuration
└── README.md            # Project documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

- **arkSong** - *Initial work* - [arksong2018@gmail.com](mailto:arksong2018@gmail.com)

## Acknowledgments

- [Ollama](https://ollama.ai/) for providing the local AI engine
- [OpenAI](https://openai.com/) for their API services
- [Deepseek](https://deepseek.com/) for their powerful AI models and API services
- The Rust community for their excellent tools and libraries

