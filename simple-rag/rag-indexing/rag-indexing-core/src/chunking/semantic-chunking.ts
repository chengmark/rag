import { Document } from '../types/Document';
import { pipeline } from '@huggingface/transformers';
import type { FeatureExtractionPipeline } from '@huggingface/transformers';
import { logger } from '../utils/logger';

export interface SemanticChunkingOptions {
  maxChunkSize?: number;
  minChunkSize?: number;
  similarityThreshold?: number;
  embeddingModel?: string;
  overlapSize?: number;
}

export interface Chunk {
  content: string;
  metadata: Record<string, string>;
  startIndex: number;
  endIndex: number;
}

export class SemanticChunker {
  private options: Required<SemanticChunkingOptions>;
  private embeddingsCache: Map<string, number[]> = new Map();
  private embeddingPipeline: FeatureExtractionPipeline | null = null;

  constructor(options: SemanticChunkingOptions = {}) {
    this.options = {
      maxChunkSize: options.maxChunkSize ?? 1000,
      minChunkSize: options.minChunkSize ?? 100,
      similarityThreshold: options.similarityThreshold ?? 0.3,
      embeddingModel: options.embeddingModel ?? 'Xenova/all-MiniLM-L6-v2',
      overlapSize: options.overlapSize ?? 50,
    };
  }

  /**
   * Split document into semantic chunks using embeddings
   */
  async chunkDocument(document: Document): Promise<Document[]> {
    const sentences = this.splitIntoSentences(document.content);
    if (sentences.length <= 1) {
      return [document];
    }

    // Generate embeddings for all sentences
    const embeddings = await this.generateEmbeddings(sentences);
    
    // Calculate semantic similarity between consecutive sentences
    const similarities = this.calculateSimilarities(embeddings);
    
    // Identify chunk boundaries based on similarity drops
    const boundaries = this.identifyBoundaries(similarities, sentences);
    
    // Create chunks based on boundaries
    const chunks = this.createChunks(sentences, boundaries, document.metadata);
    
    return chunks;
  }

  /**
   * Split text into sentences
   */
  private splitIntoSentences(text: string): string[] {
    return text
      .split(/[.!?]+/)
      .map(s => s.trim())
      .filter(s => s.length > 0);
  }

  /**
   * Initialize the embedding pipeline if not already done
   */
  private async initializePipeline(): Promise<void> {
    if (!this.embeddingPipeline) {
      try {
        const pipelineResult = await pipeline(
          'feature-extraction',
          this.options.embeddingModel,
          { device: 'cpu' }
        );
        this.embeddingPipeline = pipelineResult as FeatureExtractionPipeline;
      } catch (error) {
        logger.error('Failed to initialize embedding pipeline', error, {
          operation: 'initialize_pipeline',
          model: this.options.embeddingModel
        });
        throw new Error(`Failed to load embedding model: ${this.options.embeddingModel}`);
      }
    }
  }

  /**
   * Generate embeddings for sentences using local HuggingFace transformers
   */
  private async generateEmbeddings(sentences: string[]): Promise<number[][]> {
    await this.initializePipeline();
    
    const embeddings: number[][] = [];
    const uncachedSentences: string[] = [];
    const uncachedIndices: number[] = [];
    
    // Check cache first and collect uncached sentences
    for (let i = 0; i < sentences.length; i++) {
      const sentence = sentences[i];
      if (this.embeddingsCache.has(sentence)) {
        embeddings.push(this.embeddingsCache.get(sentence)!);
      } else {
        embeddings.push([] as number[]); // Placeholder
        uncachedSentences.push(sentence);
        uncachedIndices.push(i);
      }
    }

    // Generate embeddings for uncached sentences in batch
    if (uncachedSentences.length > 0) {
      try {
        const result = await this.embeddingPipeline!(uncachedSentences, {
          pooling: 'mean',
          normalize: true
        });
        
        const newEmbeddings = result.tolist() as number[][];
        
        // Update cache and embeddings array
        for (let i = 0; i < uncachedSentences.length; i++) {
          const sentence = uncachedSentences[i];
          const embedding = newEmbeddings[i];
          const originalIndex = uncachedIndices[i];
          
          this.embeddingsCache.set(sentence, embedding);
          embeddings[originalIndex] = embedding;
        }
      } catch (error) {
        logger.error('Error generating embeddings for batch', error, {
          operation: 'generate_embeddings',
          batchSize: uncachedSentences.length,
          model: this.options.embeddingModel
        });
        // Fallback: use zero vectors for failed embeddings
        for (const index of uncachedIndices) {
          embeddings[index] = new Array(384).fill(0); // Default dimension for MiniLM
        }
      }
    }
    
    return embeddings;
  }

  /**
   * Calculate cosine similarity between consecutive sentence embeddings
   */
  private calculateSimilarities(embeddings: number[][]): number[] {
    const similarities: number[] = [];
    
    for (let i = 0; i < embeddings.length - 1; i++) {
      const similarity = this.cosineSimilarity(embeddings[i], embeddings[i + 1]);
      similarities.push(similarity);
    }
    
    return similarities;
  }

  /**
   * Calculate cosine similarity between two vectors
   */
  private cosineSimilarity(vecA: number[], vecB: number[]): number {
    if (vecA.length !== vecB.length) {
      throw new Error('Vectors must have same dimension');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }

    if (normA === 0 || normB === 0) {
      return 0;
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Identify chunk boundaries based on similarity threshold and size constraints
   */
  private identifyBoundaries(similarities: number[], sentences: string[]): number[] {
    const boundaries: number[] = [0]; // Always start at beginning
    let currentChunkSize = 0;

    for (let i = 0; i < similarities.length; i++) {
      const sentenceLength = sentences[i].length;
      currentChunkSize += sentenceLength;

      const shouldSplit = 
        // Split if similarity is below threshold
        similarities[i] < this.options.similarityThreshold ||
        // Split if chunk exceeds max size
        currentChunkSize >= this.options.maxChunkSize;

      if (shouldSplit && 
          currentChunkSize >= this.options.minChunkSize &&
          i < sentences.length - 1) {
        boundaries.push(i + 1);
        currentChunkSize = 0;
      } else {
        currentChunkSize += 1; // Add space for separator
      }
    }

    return boundaries;
  }

  /**
   * Create document chunks based on boundaries
   */
  private createChunks(
    sentences: string[], 
    boundaries: number[], 
    originalMetadata: Record<string, string>
  ): Document[] {
    const chunks: Document[] = [];

    for (let i = 0; i < boundaries.length; i++) {
      const startIdx = boundaries[i];
      const endIdx = i < boundaries.length - 1 ? boundaries[i + 1] : sentences.length;
      
      const chunkSentences = sentences.slice(startIdx, endIdx);
      const content = chunkSentences.join('. ') + (chunkSentences.length > 0 ? '.' : '');
      
      if (content.trim().length > 0) {
        const metadata = {
          ...originalMetadata,
          chunkIndex: i.toString(),
          chunkStart: startIdx.toString(),
          chunkEnd: endIdx.toString(),
          chunkingMethod: 'semantic',
          sentenceCount: chunkSentences.length.toString(),
        };

        chunks.push({ content, metadata });
      }
    }

    return chunks;
  }

  /**
   * Get information about the loaded model
   */
  async getModelInfo(): Promise<{ model: string; loaded: boolean; cacheSize: number }> {
    return {
      model: this.options.embeddingModel,
      loaded: this.embeddingPipeline !== null,
      cacheSize: this.embeddingsCache.size
    };
  }

  /**
   * Clear the embeddings cache and optionally unload the model
   */
  clearCache(unloadModel: boolean = false): void {
    this.embeddingsCache.clear();
    if (unloadModel) {
      this.embeddingPipeline = null;
    }
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { size: number; keys: string[] } {
    return {
      size: this.embeddingsCache.size,
      keys: Array.from(this.embeddingsCache.keys()),
    };
  }
}

/**
 * Convenience function for simple semantic chunking
 */
export async function semanticChunk(
  document: Document, 
  options?: SemanticChunkingOptions
): Promise<Document[]> {
  const chunker = new SemanticChunker(options);
  return chunker.chunkDocument(document);
}