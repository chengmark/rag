# rag-indexing-core

## Features
- Document loading and parsing
    - webpage-loader
    - json-loader
    - markdown-loader
    - pdf-loader
- Text chunking
    - semantic chunking - uses huggingface embeddings to identify natural breakpoints
    - fixed size chunking - splits text into fixed size chunks
- Vector embedding generation
    - uses huggingface embeddings
- Vector storage
    - uses pinecone



## Tech Stack
- TypeScript
- Pinecone
- minimal dependencies