# RAG Retrieval Core

Core RAG retrieval functionality with semantic search, similarity matching, and query processing.

## Features

- **Semantic Search**: String-based embedding search with cosine similarity
- **Flexible Search Options**: Configurable top-K, similarity thresholds, and embedding formats
- **Batch Processing**: Multiple query support with parallel processing
- **Metadata Handling**: Preserves and returns chunk metadata
- **Statistics & Debugging**: File stats, chunk inspection, and performance metrics

## Installation

```bash
npm install @rag-retrieval/core
```

## Quick Start

```typescript
import { SearchChunkByString } from '@rag-retrieval/core';

// Initialize searcher
const searcher = new SearchChunkByString('./embeddings.json');

// Search for similar chunks
const results = await searcher.search('What is artificial intelligence?', {
  topK: 5,
  similarityThreshold: 0.3
});

console.log(results);
```

## API Reference

### SearchChunkByString

Main class for semantic search functionality.

```typescript
const searcher = new SearchChunkByString(embeddingFilePath, options);
```

#### Options

- `topK`: Number of results to return (default: 5)
- `similarityThreshold`: Minimum similarity score (default: 0.1)
- `embeddingModel`: Model for query embeddings (default: 'Xenova/all-MiniLM-L6-v2')
- `embeddingFormat`: Format for query embeddings (default: 'normalized')
- `includeMetadata`: Whether to include metadata in results (default: true)

#### Methods

- `search(query, options?)`: Search for similar chunks
- `getStats()`: Get statistics about loaded embeddings
- `getAllChunks()`: Get all chunk contents
- `reload()`: Reload embeddings from file

### Convenience Functions

```typescript
import { searchChunksByString, searchMultipleQueries, findMostSimilarChunks } from '@rag-retrieval/core';

// Quick search
const results = await searchChunksByString('machine learning', './embeddings.json');

// Batch search
const batchResults = await searchMultipleQueries(
  ['AI', 'ML', 'neural networks'], 
  './embeddings.json'
);

// Find best matches
const bestMatches = await findMostSimilarChunks(
  ['deep learning', 'computer vision'], 
  './embeddings.json'
);
```

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Run tests
npm test

# Development mode
npm run dev

# Linting
npm run lint
npm run lint:fix
```

## Dependencies

- `@rag-indexing/core`: Core indexing functionality
- `node-fetch`: HTTP client for API calls

## License

MIT
