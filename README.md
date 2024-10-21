# Embedding Service

This project implements a gRPC-based embedding service using the multilingual-e5-large model. It provides a simple and efficient way to generate embeddings for text inputs.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Docker Deployment](#docker-deployment)
- [Contributing](#contributing)
- [License](#license)

## Features

- gRPC-based embedding service
- Uses the multilingual-e5-large model for generating embeddings
- Docker support for easy deployment
- Efficient ONNX runtime for inference

## Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- Git

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/abreza/grpc-embedding-service.git
   cd grpc-embedding-service
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Generate gRPC code from the proto file:
   ```
   python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. embedding_service.proto
   ```

## Usage

To start the gRPC server:

```
python main.py
```

The server will start and listen on port 50051.

## API Reference

The service provides a single RPC method:

- `GenerateEmbedding(EmbeddingRequest) returns (EmbeddingResponse)`
  - Input: `text` (string) - The text to generate an embedding for
  - Output: `embedding` (repeated float) - The generated embedding vector

## Docker Deployment

To build and run the service using Docker:

1. Build the Docker image:
   ```
   docker-compose build
   ```

2. Start the service:
   ```
   docker-compose up
   ```

The service will be available on the host machine at `localhost:50051`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
