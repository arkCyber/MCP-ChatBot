#!/bin/bash

# Create models directory if it doesn't exist
mkdir -p models

# Download BERT tokenizer
echo "Downloading BERT tokenizer..."
curl -L "https://huggingface.co/bert-base-uncased/raw/main/tokenizer.json" -o models/tokenizer.json

echo "Model files downloaded successfully!" 