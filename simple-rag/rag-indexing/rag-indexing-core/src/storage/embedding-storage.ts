import { writeFileSync, readFileSync, existsSync } from 'fs';
import { Embedding, EmbeddingFormat } from '../embedding/convert-embedding';
import { Document } from '../types/Document';
import { logger } from '../utils/logger';

/**
 * Embedding storage interface for different backends
 */
export interface EmbeddingStorage {
  save(embeddings: StoredEmbedding[]): Promise<void>;
  load(): Promise<StoredEmbedding[]>;
  search(queryEmbedding: number[], topK: number): Promise<EmbeddingSearchResult[]>;
}

/**
 * Stored embedding with metadata
 */
export interface StoredEmbedding {
  id: string;
  embedding: number[];
  content: string;
  metadata: Record<string, string>;
  format: EmbeddingFormat;
  model: string;
  dimensions: number;
  timestamp: number;
}

/**
 * Search result with similarity score
 */
export interface EmbeddingSearchResult {
  embedding: StoredEmbedding;
  similarity: number;
  content: string;
}

/**
 * File-based embedding storage using JSON format
 */
export class FileEmbeddingStorage implements EmbeddingStorage {
  private filePath: string;

  constructor(filePath: string = './embeddings.json') {
    this.filePath = filePath;
  }

  /**
   * Save embeddings to JSON file
   */
  async save(embeddings: StoredEmbedding[]): Promise<void> {
    try {
      const data = {
        version: '1.0.0',
        created: new Date().toISOString(),
        count: embeddings.length,
        embeddings: embeddings
      };
      
      writeFileSync(this.filePath, JSON.stringify(data, null, 2), 'utf-8');
      logger.info(`Saved ${embeddings.length} embeddings to ${this.filePath}`, {
        operation: 'save_embeddings',
        filePath: this.filePath,
        embeddingCount: embeddings.length
      });
    } catch (error) {
      logger.error('Failed to save embeddings', error, {
        operation: 'save_embeddings',
        filePath: this.filePath,
        embeddingCount: embeddings.length
      });
      throw new Error(`Failed to save embeddings: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Load embeddings from JSON file
   */
  async load(): Promise<StoredEmbedding[]> {
    if (!existsSync(this.filePath)) {
      return [];
    }

    try {
      const data = JSON.parse(readFileSync(this.filePath, 'utf-8'));
      const embeddingCount = data.embeddings?.length || 0;
      logger.info(`Loaded ${embeddingCount} embeddings from ${this.filePath}`, {
        operation: 'load_embeddings',
        filePath: this.filePath,
        embeddingCount
      });
      return data.embeddings || [];
    } catch (error) {
      logger.error('Failed to load embeddings', error, {
        operation: 'load_embeddings',
        filePath: this.filePath
      });
      throw new Error(`Failed to load embeddings: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Search for similar embeddings using cosine similarity
   */
  async search(queryEmbedding: number[], topK: number = 5): Promise<EmbeddingSearchResult[]> {
    const embeddings = await this.load();
    
    if (embeddings.length === 0) {
      return [];
    }

    // Calculate similarities
    const results: EmbeddingSearchResult[] = embeddings.map(stored => ({
      embedding: stored,
      similarity: this.cosineSimilarity(queryEmbedding, stored.embedding),
      content: stored.content
    }));

    // Sort by similarity (descending) and take top K
    return results
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);
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
   * Get storage statistics
   */
  async getStats(): Promise<{
    fileExists: boolean;
    fileSize: number;
    embeddingCount: number;
    lastModified?: string;
  }> {
    const fileExists = existsSync(this.filePath);
    
    if (!fileExists) {
      return {
        fileExists: false,
        fileSize: 0,
        embeddingCount: 0
      };
    }

    try {
      const stats = require('fs').statSync(this.filePath);
      const data = JSON.parse(readFileSync(this.filePath, 'utf-8'));
      
      return {
        fileExists: true,
        fileSize: stats.size,
        embeddingCount: data.embeddings?.length || 0,
        lastModified: stats.mtime.toISOString()
      };
    } catch (error) {
      return {
        fileExists: true,
        fileSize: 0,
        embeddingCount: 0
      };
    }
  }
}

/**
 * Convert Document chunks to StoredEmbedding format
 */
export function documentChunksToStoredEmbeddings(
  chunks: Document[],
  embeddings: Embedding[]
): StoredEmbedding[] {
  if (chunks.length !== embeddings.length) {
    throw new Error('Chunks and embeddings must have the same length');
  }

  return chunks.map((chunk, index) => ({
    id: `chunk_${index}_${Date.now()}`,
    embedding: embeddings[index].vector,
    content: chunk.content,
    metadata: {
      ...chunk.metadata,
      embeddingFormat: embeddings[index].format,
      embeddingModel: embeddings[index].model,
      chunkIndex: index.toString()
    },
    format: embeddings[index].format,
    model: embeddings[index].model,
    dimensions: embeddings[index].dimensions,
    timestamp: embeddings[index].timestamp
  }));
}

/**
 * Convenience function to save embeddings from chunks
 */
export async function saveChunkEmbeddings(
  chunks: Document[],
  embeddings: Embedding[],
  filePath?: string
): Promise<void> {
  const storage = new FileEmbeddingStorage(filePath);
  const storedEmbeddings = documentChunksToStoredEmbeddings(chunks, embeddings);
  await storage.save(storedEmbeddings);
}

/**
 * Convenience function to search embeddings
 */
export async function searchEmbeddings(
  queryEmbedding: number[],
  topK: number = 5,
  filePath?: string
): Promise<EmbeddingSearchResult[]> {
  const storage = new FileEmbeddingStorage(filePath);
  return storage.search(queryEmbedding, topK);
}
