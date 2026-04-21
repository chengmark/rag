import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import { pipeline } from '@huggingface/transformers';
import { logger } from '../utils/logger';

// Local type definitions (matching rag-indexing-core)
export enum EmbeddingFormat {
  RAW = 'raw',
  NORMALIZED = 'normalized',
  BINARY = 'binary',
  QUANTIZED = 'quantized'
}

// HuggingFace pipeline for embedding generation
let embeddingPipeline: any = null;

/**
 * Initialize the HuggingFace embedding pipeline
 */
async function initializeEmbeddingPipeline(model: string = 'Xenova/all-MiniLM-L6-v2') {
  if (!embeddingPipeline) {
    embeddingPipeline = await pipeline(
      'feature-extraction',
      model,
      { device: 'cpu' }
    );
  }
  return embeddingPipeline;
}

/**
 * Generate embedding using HuggingFace pipeline (matching rag-indexing-core approach)
 */
async function generateEmbedding(
  text: string,
  model: string = 'Xenova/all-MiniLM-L6-v2',
  options: { targetFormat: EmbeddingFormat } = { targetFormat: EmbeddingFormat.NORMALIZED }
): Promise<{ vector: number[]; dimensions: number; format: EmbeddingFormat; model: string }> {
  await initializeEmbeddingPipeline(model);
  
  const result = await embeddingPipeline!(text, {
    pooling: 'mean',
    normalize: false // We'll handle normalization ourselves
  });
  
  let vector = result.tolist() as number[];
  
  // Handle the case where result might be nested (for single text input)
  if (Array.isArray(vector[0])) {
    vector = vector[0] as number[];
  }
  
  logger.debug(`Generated embedding dimensions: ${vector.length}`);
  
  let processedVector = [...vector];
  
  // Apply format conversions
  if (options.targetFormat === EmbeddingFormat.NORMALIZED) {
    const norm = Math.sqrt(processedVector.reduce((sum, val) => sum + val * val, 0));
    if (norm > 0) {
      processedVector = processedVector.map(val => val / norm);
    }
  }
  
  return {
    vector: processedVector,
    dimensions: processedVector.length,
    format: options.targetFormat,
    model
  };
}

/**
 * Search result with similarity score and metadata
 */
export interface SearchResult {
  content: string;
  similarity: number;
  metadata: Record<string, string>;
  chunkIndex: string;
  source: string;
}

/**
 * Search options for string-based embedding search
 */
export interface SearchOptions {
  topK?: number;
  similarityThreshold?: number;
  embeddingModel?: string;
  embeddingFormat?: EmbeddingFormat;
  includeMetadata?: boolean;
}

/**
 * Default search options
 */
const DEFAULT_SEARCH_OPTIONS: Required<SearchOptions> = {
  topK: 5,
  similarityThreshold: 0.1,
  embeddingModel: 'Xenova/all-MiniLM-L6-v2',
  embeddingFormat: EmbeddingFormat.NORMALIZED,
  includeMetadata: true
};

/**
 * Search chunks by string query using embedding similarity
 */
export class SearchChunkByString {
  private embeddingFilePath: string;
  private options: Required<SearchOptions>;
  private embeddings: any[] = [];

  constructor(embeddingFilePath: string, options: SearchOptions = {}) {
    this.embeddingFilePath = embeddingFilePath;
    this.options = { ...DEFAULT_SEARCH_OPTIONS, ...options };
  }

  /**
   * Load embeddings from JSON file
   */
  private loadEmbeddings(): void {
    if (!existsSync(this.embeddingFilePath)) {
      throw new Error(`Embedding file not found: ${this.embeddingFilePath}`);
    }

    try {
      const data = JSON.parse(readFileSync(this.embeddingFilePath, 'utf-8'));
      this.embeddings = data.embeddings || [];
      logger.info(`Loaded ${this.embeddings.length} embeddings from ${this.embeddingFilePath}`);
    } catch (error) {
      logger.error('Failed to load embeddings', error);
      throw new Error(`Failed to load embeddings: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Search for similar chunks using string query
   */
  async search(query: string, options?: Partial<SearchOptions>): Promise<SearchResult[]> {
    const searchOptions = { ...this.options, ...options };
    
    // Load embeddings if not already loaded
    if (this.embeddings.length === 0) {
      this.loadEmbeddings();
    }

    if (this.embeddings.length === 0) {
      return [];
    }

    try {
      // Generate embedding for the query
      logger.info(`Generating embedding for query: "${query}"`);
      const queryEmbedding = await generateEmbedding(query, searchOptions.embeddingModel, {
        targetFormat: searchOptions.embeddingFormat
      });

      // Calculate similarities
      const results: SearchResult[] = this.embeddings.map((storedEmbedding, index) => {
        const similarity = this.cosineSimilarity(
          (queryEmbedding as any).vector,
          storedEmbedding.embedding
        );

        return {
          content: storedEmbedding.content,
          similarity,
          metadata: searchOptions.includeMetadata ? storedEmbedding.metadata : {},
          chunkIndex: storedEmbedding.metadata?.chunkIndex || index.toString(),
          source: storedEmbedding.metadata?.source || 'unknown'
        };
      });

      // Filter by similarity threshold and sort by similarity (descending)
      return results
        .filter(result => result.similarity >= searchOptions.similarityThreshold)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, searchOptions.topK);

    } catch (error) {
      logger.error('Search failed', error);
      throw new Error(`Search failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Calculate cosine similarity between two vectors
  */
  private cosineSimilarity(vecA: number[], vecB: number[]): number {
    if (vecA.length !== vecB.length) {
      throw new Error('Vectors must have same dimensions');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Get statistics about the loaded embeddings
   */
  getStats(): {
    fileExists: boolean;
    embeddingCount: number;
    dimensions: number;
    model: string;
    format: string;
  } {
    const fileExists = existsSync(this.embeddingFilePath);
    
    if (!fileExists) {
      return {
        fileExists: false,
        embeddingCount: 0,
        dimensions: 0,
        model: '',
        format: ''
      };
    }

    if (this.embeddings.length === 0) {
      this.loadEmbeddings();
    }

    const firstEmbedding = this.embeddings[0];
    return {
      fileExists: true,
      embeddingCount: this.embeddings.length,
      dimensions: firstEmbedding?.embedding?.length || 0,
      model: firstEmbedding?.metadata?.embeddingModel || 'unknown',
      format: firstEmbedding?.metadata?.embeddingFormat || 'unknown'
    };
  }

  /**
   * Reload embeddings from file
   */
  reload(): void {
    this.embeddings = [];
    this.loadEmbeddings();
  }

  /**
   * Get all chunk contents (for debugging or inspection)
   */
  getAllChunks(): Array<{ content: string; metadata: Record<string, string> }> {
    if (this.embeddings.length === 0) {
      this.loadEmbeddings();
    }

    return this.embeddings.map(embedding => ({
      content: embedding.content,
      metadata: embedding.metadata || {}
    }));
  }
}

/**
 * Convenience function for quick search
 */
export async function searchChunksByString(
  query: string,
  embeddingFilePath: string,
  options?: SearchOptions
): Promise<SearchResult[]> {
  const searcher = new SearchChunkByString(embeddingFilePath, options);
  return searcher.search(query, options);
}

/**
 * Convenience function for multiple queries
 */
export async function searchMultipleQueries(
  queries: string[],
  embeddingFilePath: string,
  options?: SearchOptions
): Promise<Array<{ query: string; results: SearchResult[] }>> {
  const searcher = new SearchChunkByString(embeddingFilePath, options);
  
  const results = await Promise.all(
    queries.map(async query => ({
      query,
      results: await searcher.search(query, options)
    }))
  );

  return results;
}

/**
 * Find most similar chunks across multiple queries
 */
export async function findMostSimilarChunks(
  queries: string[],
  embeddingFilePath: string,
  options?: SearchOptions
): Promise<Array<{ query: string; bestMatch: SearchResult; allResults: SearchResult[] }>> {
  const searcher = new SearchChunkByString(embeddingFilePath, options);
  
  const results = await Promise.all(
    queries.map(async query => {
      const allResults = await searcher.search(query, options);
      const bestMatch = allResults.length > 0 ? allResults[0] : null;
      
      return {
        query,
        bestMatch,
        allResults
      };
    })
  );

  return results.filter(result => result.bestMatch !== null) as Array<{
    query: string;
    bestMatch: SearchResult;
    allResults: SearchResult[];
  }>;
}