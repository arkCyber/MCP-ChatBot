#!/bin/bash

set -e  # Exit on error

# Check if we're in the right directory
if [ ! -d "qdrant" ]; then
    echo "Error: Qdrant source directory not found"
    exit 1
fi

cd qdrant

# Check Rust version
RUST_VERSION=$(rustc --version | cut -d' ' -f2)
echo "Using Rust version: $RUST_VERSION"

# Set environment variables for the build
export OPENSSL_STATIC=1
export OPENSSL_DIR=$(brew --prefix openssl@3)
export PROTOC=$(which protoc)
export PROTOC_INCLUDE=$(brew --prefix protobuf)/include

echo "Building Qdrant..."
echo "Using OpenSSL from: $OPENSSL_DIR"
echo "Using protoc from: $PROTOC"

# Clean any previous build artifacts
cargo clean

# Build with release profile and specific features
RUSTFLAGS="-C target-cpu=native" cargo build --release --no-default-features --features "default"

# Create necessary directories
mkdir -p ../data/qdrant
mkdir -p ../config

# Create config file
cat > ../config/config.json << EOF
{
    "service": {
        "host": "0.0.0.0",
        "port": 6333,
        "http_port": 6334
    },
    "storage": {
        "storage_path": "../data/qdrant"
    },
    "optimizers": {
        "default_segment_number": 2,
        "memmap_threshold": 20000
    },
    "wal": {
        "wal_capacity_mb": 32,
        "wal_segments_capacity_mb": 64
    },
    "performance": {
        "max_search_threads": 4,
        "max_optimization_threads": 2
    }
}
EOF

echo "Build completed successfully!"
echo "Qdrant binary is located at: target/release/qdrant"
echo "Configuration file is at: ../config/config.json" 