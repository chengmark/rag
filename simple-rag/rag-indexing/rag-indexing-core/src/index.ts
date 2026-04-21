// Main exports for rag-indexing-core package

// Types
export { Document } from './types/Document';

// Text loading
export {
  TxtLoader,
  loadTxt,
  loadMultipleTxt,
  loadTxtFromString,
  type TxtLoaderOptions
} from './loaders/txt-loader';

// Semantic chunking
export { 
  SemanticChunker, 
  semanticChunk,
  type SemanticChunkingOptions,
  type Chunk
} from './chunking/semantic-chunking';

// Embedding conversion
export {
  EmbeddingConverter,
  convertEmbedding,
  generateEmbedding,
  type Embedding,
  type ConversionOptions,
  EmbeddingFormat
} from './embedding/convert-embedding';

// Embedding storage
export {
  FileEmbeddingStorage,
  saveChunkEmbeddings,
  searchEmbeddings,
  documentChunksToStoredEmbeddings,
  type EmbeddingStorage,
  type StoredEmbedding,
  type EmbeddingSearchResult
} from './storage/embedding-storage';

// Package version
export const VERSION = '1.0.0';
