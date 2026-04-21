// Main exports for rag-retrieval-core package

// Search functionality
export {
  SearchChunkByString,
  searchChunksByString,
  searchMultipleQueries,
  findMostSimilarChunks,
  type SearchResult,
  type SearchOptions
} from './retrieve/search-chunk-by-string';

// Types and enums
export { EmbeddingFormat } from './retrieve/search-chunk-by-string';

// Package version
export const VERSION = '1.0.0';
