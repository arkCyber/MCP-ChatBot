#!/bin/bash

# 设置错误时退出
set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
QDRANT_IMAGE="qdrant/qdrant:latest"
QDRANT_CONTAINER="qdrant"
QDRANT_HTTP_PORT=6334
QDRANT_GRPC_PORT=6333
STORAGE_DIR="./qdrant_storage"

# 打印带颜色的消息
print_message() {
    echo -e "${2}${1}${NC}"
}

# 检查 Docker 是否运行
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_message "Error: Docker is not running. Please start Docker first." "$RED"
        exit 1
    fi
}

# 创建存储目录
create_storage_dir() {
    if [ ! -d "$STORAGE_DIR" ]; then
        print_message "Creating storage directory: $STORAGE_DIR" "$YELLOW"
        mkdir -p "$STORAGE_DIR"
    fi
}

# 拉取 Qdrant 镜像
pull_qdrant_image() {
    print_message "Pulling Qdrant image..." "$YELLOW"
    if ! docker pull $QDRANT_IMAGE; then
        print_message "Error: Failed to pull Qdrant image." "$RED"
        exit 1
    fi
    print_message "Qdrant image pulled successfully." "$GREEN"
}

# 启动 Qdrant 容器
start_qdrant() {
    # 检查容器是否已存在
    if docker ps -a | grep -q $QDRANT_CONTAINER; then
        print_message "Stopping existing Qdrant container..." "$YELLOW"
        docker stop $QDRANT_CONTAINER > /dev/null 2>&1 || true
        docker rm $QDRANT_CONTAINER > /dev/null 2>&1 || true
    fi

    print_message "Starting Qdrant container..." "$YELLOW"
    docker run -d \
        --name $QDRANT_CONTAINER \
        -p $QDRANT_HTTP_PORT:$QDRANT_HTTP_PORT \
        -p $QDRANT_GRPC_PORT:$QDRANT_GRPC_PORT \
        -v "$(pwd)/$STORAGE_DIR:/qdrant/storage" \
        $QDRANT_IMAGE

    # 等待容器启动
    print_message "Waiting for Qdrant to start..." "$YELLOW"
    sleep 5

    # 检查容器是否正常运行
    if ! docker ps | grep -q $QDRANT_CONTAINER; then
        print_message "Error: Failed to start Qdrant container." "$RED"
        exit 1
    fi

    print_message "Qdrant is running successfully!" "$GREEN"
    print_message "HTTP API: http://localhost:$QDRANT_HTTP_PORT" "$GREEN"
    print_message "gRPC API: localhost:$QDRANT_GRPC_PORT" "$GREEN"
}

# 主函数
main() {
    print_message "Setting up Qdrant..." "$YELLOW"
    
    check_docker
    create_storage_dir
    pull_qdrant_image
    start_qdrant
    
    print_message "Setup completed successfully!" "$GREEN"
}

# 运行主函数
main 