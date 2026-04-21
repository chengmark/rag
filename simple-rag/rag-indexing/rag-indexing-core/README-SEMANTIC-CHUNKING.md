# Semantic Chunking with Local Hugging Face Transformers

This document explains how to use the updated semantic chunking implementation that runs Hugging Face sentence transformers locally instead of using API calls.

## Installation

First, install the required dependencies:

```bash
npm install @huggingface/transformers@3.0.0
```

## Key Changes

### Before (API-based)
- Used external Hugging Face API calls
- Required internet connection for each request
- Had API rate limits and potential costs
- Used model: `sentence-transformers/all-MiniLM-L6-v2`

### After (Local Transformers)
- Uses local ONNX models via Transformers.js
- Models are downloaded once and cached locally
- Works offline after initial download
- Uses model: `Xenova/all-MiniLM-L6-v2` (ONNX version)

## Usage

### Basic Usage

```typescript
import { SemanticChunker, Document } from './src/chunking/semantic-chunking';

const document: Document = {
  content: "Your long text content here...",
  metadata: { source: "example" }
};

const chunker = new SemanticChunker({
  maxChunkSize: 300,
  similarityThreshold: 0.3,
  embeddingModel: 'Xenova/all-MiniLM-L6-v2'
});

const chunks = await chunker.chunkDocument(document);
console.log(`Generated ${chunks.length} semantic chunks`);
```

### Convenience Function

```typescript
import { semanticChunk, Document } from './src/chunking/semantic-chunking';

const chunks = await semanticChunk(document, {
  maxChunkSize: 200,
  similarityThreshold: 0.4
});
```

## Available Models

### Recommended Models

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| `Xenova/all-MiniLM-L6-v2` | ~1.2GB | Fast | General purpose, CPU-friendly |
| `Xenova/bge-m3` | ~2.1GB | Medium | Multi-retrieval (dense, sparse, colbert) |
| `Xenova/e5-small` | ~2.0GB | Fast | High accuracy retrieval |
| `mixedbread-ai/mxbai-embed-xsmall-v1` | Very small | Very Fast | Resource-constrained environments |

### Model Selection

```typescript
// For resource-constrained environments
const chunker = new SemanticChunker({
  embeddingModel: 'Xenova/all-MiniLM-L6-v2'
});

// For higher accuracy
const chunker = new SemanticChunker({
  embeddingModel: 'Xenova/e5-small'
});

// For multi-retrieval capabilities
const chunker = new SemanticChunker({
  embeddingModel: 'Xenova/bge-m3'
});
```

## Configuration Options

```typescript
interface SemanticChunkingOptions {
  maxChunkSize?: number;        // Maximum characters per chunk (default: 1000)
  minChunkSize?: number;        // Minimum characters per chunk (default: 100)
  similarityThreshold?: number; // Similarity threshold for splitting (default: 0.3)
  embeddingModel?: string;      // Model name (default: 'Xenova/all-MiniLM-L6-v2')
  overlapSize?: number;         // Overlap between chunks (default: 50)
}
```

## Performance Features

### Caching
- Sentence embeddings are cached automatically
- Reduces computation for repeated sentences
- Cache can be cleared with `clearCache()`

### Batch Processing
- Embeddings are generated in batches for efficiency
- Uncached sentences are processed together
- Cached sentences are skipped

### Model Management
```typescript
// Get model information
const info = await chunker.getModelInfo();
console.log(info); // { model: 'Xenova/all-MiniLM-L6-v2', loaded: true, cacheSize: 42 }

// Clear cache but keep model loaded
chunker.clearCache();

// Clear cache and unload model
chunker.clearCache(true);
```

## Running the Example

```bash
# Install dependencies first
npm install

# Run the example
npx ts-node examples/semantic-chunking-example.ts
```

## Troubleshooting

### Common Issues

1. **Module not found error**
   ```bash
   npm install @huggingface/transformers@3.0.0
   ```

2. **Memory issues**
   - Try smaller models: `Xenova/all-MiniLM-L6-v2`
   - Reduce batch size in your code
   - Ensure sufficient RAM is available

3. **Model download fails**
   - Check internet connection for first download
   - Models are cached after first successful download
   - Try different model if download fails

4. **TypeScript errors**
   - Ensure `@huggingface/transformers` is installed
   - Check TypeScript version compatibility
   - Use provided type imports

### Performance Tips

1. **Use appropriate model size**
   - Smaller models for CPU deployment
   - Larger models for better accuracy if resources allow

2. **Optimize chunk sizes**
   - Larger chunks = fewer embeddings to compute
   - Smaller chunks = better semantic precision

3. **Leverage caching**
   - Reuse chunker instances for multiple documents
   - Clear cache only when necessary

## Migration from API Version

If you're migrating from the API-based version:

1. Update dependencies:
   ```bash
   npm install @huggingface/transformers@3.0.0
   ```

2. Change model names:
   ```typescript
   // Before
   embeddingModel: 'sentence-transformers/all-MiniLM-L6-v2'
   
   // After
   embeddingModel: 'Xenova/all-MiniLM-L6-v2'
   ```

3. Remove API keys (no longer needed)

4. Update imports if needed:
   ```typescript
   import { SemanticChunker } from './src/chunking/semantic-chunking';
   ```

## Browser Support

The same code works in browsers with WebGPU support:

```typescript
const chunker = new SemanticChunker({
  embeddingModel: 'Xenova/all-MiniLM-L6-v2'
});

// For WebGPU acceleration (if supported)
const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
  device: 'webgpu'
});
```

## License

This implementation uses Hugging Face Transformers.js, which is licensed under Apache 2.0.
