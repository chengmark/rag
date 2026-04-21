import { pipeline, type FeatureExtractionPipeline } from '@huggingface/transformers';

/**
 * Supported embedding formats
 */
export enum EmbeddingFormat {
  RAW = 'raw',
  NORMALIZED = 'normalized',
  BINARY = 'binary',
  QUANTIZED = 'quantized'
}

/**
 * Embedding vector with metadata
 */
export interface Embedding {
  vector: number[];
  dimensions: number;
  format: EmbeddingFormat;
  model: string;
  timestamp: number;
}

/**
 * Embedding conversion options
 */
export interface ConversionOptions {
  targetFormat?: EmbeddingFormat;
  normalize?: boolean;
  quantize?: {
    bits: 8 | 16;
    range?: [number, number];
  };
  binary?: {
    threshold?: number;
  };
}

/**
 * Utility class for converting between different embedding formats
 */
export class EmbeddingConverter {
  private pipeline: FeatureExtractionPipeline | null = null;
  private model: string;

  constructor(model: string = 'Xenova/all-MiniLM-L6-v2') {
    this.model = model;
  }

  /**
   * Initialize the embedding pipeline
   */
  private async initializePipeline(): Promise<void> {
    if (!this.pipeline) {
      const pipelineResult = await pipeline(
        'feature-extraction',
        this.model,
        { device: 'cpu' }
      );
      this.pipeline = pipelineResult as FeatureExtractionPipeline;
    }
  }

  /**
   * Convert embedding from one format to another
   */
  async convertEmbedding(
    embedding: Embedding,
    options: ConversionOptions = {}
  ): Promise<Embedding> {
    const targetFormat = options.targetFormat || embedding.format;
    
    let vector = [...embedding.vector];
    
    // Apply conversions based on target format
    switch (targetFormat) {
      case EmbeddingFormat.NORMALIZED:
        vector = this.normalizeVector(vector);
        break;
        
      case EmbeddingFormat.BINARY:
        vector = this.toBinary(vector, options.binary?.threshold);
        break;
        
      case EmbeddingFormat.QUANTIZED:
        vector = this.quantizeVector(vector, options.quantize);
        break;
        
      case EmbeddingFormat.RAW:
        // Keep as-is, just ensure it's not normalized
        if (embedding.format === EmbeddingFormat.NORMALIZED) {
          // Could denormalize if needed, but for now keep as-is
        }
        break;
    }

    return {
      vector,
      dimensions: vector.length,
      format: targetFormat,
      model: embedding.model,
      timestamp: Date.now()
    };
  }

  /**
   * Convert multiple embeddings in batch
   */
  async convertEmbeddings(
    embeddings: Embedding[],
    options: ConversionOptions = {}
  ): Promise<Embedding[]> {
    return Promise.all(
      embeddings.map(embedding => this.convertEmbedding(embedding, options))
    );
  }

  /**
   * Generate embedding from text and convert to target format
   */
  async generateAndConvert(
    text: string | string[],
    options: ConversionOptions = {}
  ): Promise<Embedding | Embedding[]> {
    await this.initializePipeline();
    
    const isArray = Array.isArray(text);
    const texts = isArray ? text : [text];
    
    const result = await this.pipeline!(texts, {
      pooling: 'mean',
      normalize: false // We'll handle normalization ourselves
    });
    
    const vectors = result.tolist() as number[][];
    
    const embeddings: Embedding[] = vectors.map(vector => ({
      vector,
      dimensions: vector.length,
      format: EmbeddingFormat.RAW,
      model: this.model,
      timestamp: Date.now()
    }));

    if (options.targetFormat || options.normalize || options.quantize || options.binary) {
      const convertedEmbeddings = await this.convertEmbeddings(embeddings, options);
      return isArray ? convertedEmbeddings : convertedEmbeddings[0];
    }

    return isArray ? embeddings : embeddings[0];
  }

  /**
   * Normalize vector to unit length
   */
  private normalizeVector(vector: number[]): number[] {
    const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    if (norm === 0) return vector;
    return vector.map(val => val / norm);
  }

  /**
   * Convert vector to binary representation
   */
  private toBinary(vector: number[], threshold: number = 0): number[] {
    return vector.map(val => val > threshold ? 1 : 0);
  }

  /**
   * Quantize vector to specified bit depth
   */
  private quantizeVector(
    vector: number[], 
    options?: { bits: 8 | 16; range?: [number, number] }
  ): number[] {
    const bits = options?.bits || 8;
    const [min, max] = options?.range || this.findMinMax(vector);
    
    if (bits === 8) {
      return vector.map(val => {
        const normalized = (val - min) / (max - min);
        return Math.round(normalized * 255);
      });
    } else {
      return vector.map(val => {
        const normalized = (val - min) / (max - min);
        return Math.round(normalized * 65535);
      });
    }
  }

  /**
   * Find min and max values in vector
   */
  private findMinMax(vector: number[]): [number, number] {
    let min = vector[0];
    let max = vector[0];
    
    for (const val of vector) {
      if (val < min) min = val;
      if (val > max) max = val;
    }
    
    return [min, max];
  }

  /**
   * Calculate similarity between two embeddings
   */
  calculateSimilarity(
    embedding1: Embedding, 
    embedding2: Embedding,
    metric: 'cosine' | 'euclidean' | 'manhattan' = 'cosine'
  ): number {
    if (embedding1.dimensions !== embedding2.dimensions) {
      throw new Error('Embeddings must have same dimensions');
    }

    const vec1 = embedding1.vector;
    const vec2 = embedding2.vector;

    switch (metric) {
      case 'cosine':
        return this.cosineSimilarity(vec1, vec2);
      case 'euclidean':
        return this.euclideanDistance(vec1, vec2);
      case 'manhattan':
        return this.manhattanDistance(vec1, vec2);
      default:
        throw new Error(`Unknown metric: ${metric}`);
    }
  }

  /**
   * Cosine similarity calculation
   */
  private cosineSimilarity(vec1: number[], vec2: number[]): number {
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < vec1.length; i++) {
      dotProduct += vec1[i] * vec2[i];
      norm1 += vec1[i] * vec1[i];
      norm2 += vec2[i] * vec2[i];
    }

    if (norm1 === 0 || norm2 === 0) return 0;
    return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
  }

  /**
   * Euclidean distance calculation
   */
  private euclideanDistance(vec1: number[], vec2: number[]): number {
    let sum = 0;
    for (let i = 0; i < vec1.length; i++) {
      const diff = vec1[i] - vec2[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  /**
   * Manhattan distance calculation
   */
  private manhattanDistance(vec1: number[], vec2: number[]): number {
    return vec1.reduce((sum, val, i) => sum + Math.abs(val - vec2[i]), 0);
  }

  /**
   * Get format information
   */
  static getFormatInfo(format: EmbeddingFormat): {
    name: string;
    description: string;
    sizeMultiplier: number;
  } {
    switch (format) {
      case EmbeddingFormat.RAW:
        return {
          name: 'Raw',
          description: 'Original floating-point values',
          sizeMultiplier: 1
        };
      case EmbeddingFormat.NORMALIZED:
        return {
          name: 'Normalized',
          description: 'Unit-length vectors',
          sizeMultiplier: 1
        };
      case EmbeddingFormat.BINARY:
        return {
          name: 'Binary',
          description: 'Binary representation (0/1)',
          sizeMultiplier: 0.125
        };
      case EmbeddingFormat.QUANTIZED:
        return {
          name: 'Quantized',
          description: 'Reduced precision integers',
          sizeMultiplier: 0.5
        };
      default:
        throw new Error(`Unknown format: ${format}`);
    }
  }

  /**
   * Clean up resources
   */
  cleanup(): void {
    this.pipeline = null;
  }
}

/**
 * Convenience function for quick embedding conversion
 */
export async function convertEmbedding(
  embedding: Embedding,
  options: ConversionOptions = {}
): Promise<Embedding> {
  const converter = new EmbeddingConverter(embedding.model);
  try {
    return await converter.convertEmbedding(embedding, options);
  } finally {
    converter.cleanup();
  }
}

/**
 * Convenience function for generating and converting embeddings
 */
export async function generateEmbedding(
  text: string | string[],
  model: string = 'Xenova/all-MiniLM-L6-v2',
  options: ConversionOptions = {}
): Promise<Embedding | Embedding[]> {
  const converter = new EmbeddingConverter(model);
  try {
    return await converter.generateAndConvert(text, options);
  } finally {
    converter.cleanup();
  }
}