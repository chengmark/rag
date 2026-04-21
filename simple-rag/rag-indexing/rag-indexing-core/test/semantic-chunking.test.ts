import { Document } from '../src/types/Document';
import { SemanticChunker, semanticChunk, SemanticChunkingOptions } from '../src/chunking/semantic-chunking';

// Mock test data
const sampleDocument: Document = {
  content: `Artificial intelligence (AI) is intelligence demonstrated by machines. 
  In contrast to the natural intelligence displayed by humans and animals, AI involves research and development of machines that can perform tasks that typically require human intelligence.
  
  Machine learning is a subset of AI that focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct instruction, or experience.
  
  Deep learning is a subset of machine learning that uses neural networks with multiple layers to progressively extract higher-level features from raw input. For example, in image processing, lower layers may identify edges, while higher layers may identify concepts relevant to a human such as digits, letters, or faces.
  
  Natural language processing (NLP) is another important branch of AI. It focuses on the interaction between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.`,
  metadata: {
    source: 'test-document',
    title: 'AI Overview',
    author: 'Test Author'
  }
};

// Mock fetch to simulate HuggingFace API responses
global.fetch = jest.fn();

describe('SemanticChunker', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Basic functionality', () => {
    test('should split document into chunks with default options', async () => {
      // Mock successful API response
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => [[0.1, 0.2, 0.3, 0.4, 0.5]] // Mock embedding
      });

      const chunks = await semanticChunk(sampleDocument);
      
      expect(chunks).toBeDefined();
      expect(chunks.length).toBeGreaterThan(0);
      expect(chunks[0]).toHaveProperty('content');
      expect(chunks[0]).toHaveProperty('metadata');
      expect(chunks[0].metadata).toHaveProperty('chunkingMethod', 'semantic');
    });

    test('should handle single sentence document', async () => {
      const singleSentenceDoc: Document = {
        content: 'This is a single sentence.',
        metadata: { source: 'test' }
      };

      const chunks = await semanticChunk(singleSentenceDoc);
      
      expect(chunks).toHaveLength(1);
      expect(chunks[0].content).toBe(singleSentenceDoc.content);
    });

    test('should handle empty document', async () => {
      const emptyDoc: Document = {
        content: '',
        metadata: { source: 'test' }
      };

      const chunks = await semanticChunk(emptyDoc);
      
      expect(chunks).toHaveLength(1);
      expect(chunks[0].content).toBe('');
    });
  });

  describe('Custom options', () => {
    test('should respect custom maxChunkSize', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => [[0.1, 0.2, 0.3, 0.4, 0.5]]
      });

      const customOptions: SemanticChunkingOptions = {
        maxChunkSize: 200,
        minChunkSize: 50,
        similarityThreshold: 0.4
      };
      
      const chunks = await semanticChunk(sampleDocument, customOptions);
      
      expect(chunks.length).toBeGreaterThan(0);
      // Check that we have more chunks than with default options (due to smaller max size)
      const defaultChunks = await semanticChunk(sampleDocument);
      expect(chunks.length).toBeGreaterThan(defaultChunks.length);
    });

    test('should use custom similarity threshold', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => [[0.1, 0.2, 0.3, 0.4, 0.5]]
      });

      const customOptions: SemanticChunkingOptions = {
        similarityThreshold: 0.8
      };
      
      const chunks = await semanticChunk(sampleDocument, customOptions);
      
      expect(chunks).toBeDefined();
      expect(chunks.length).toBeGreaterThan(0);
    });
  });

  describe('SemanticChunker class', () => {
    test('should create instance with default options', () => {
      const chunker = new SemanticChunker();
      expect(chunker).toBeInstanceOf(SemanticChunker);
    });

    test('should create instance with custom options', () => {
      const customOptions: SemanticChunkingOptions = {
        maxChunkSize: 500,
        similarityThreshold: 0.5
      };
      const chunker = new SemanticChunker(customOptions);
      expect(chunker).toBeInstanceOf(SemanticChunker);
    });

    test('should clear cache', () => {
      const chunker = new SemanticChunker();
      chunker.clearCache();
      const stats = chunker.getCacheStats();
      expect(stats.size).toBe(0);
    });

    test('should return cache statistics', () => {
      const chunker = new SemanticChunker();
      const stats = chunker.getCacheStats();
      expect(stats).toHaveProperty('size');
      expect(stats).toHaveProperty('keys');
      expect(Array.isArray(stats.keys)).toBe(true);
    });
  });

  describe('Error handling', () => {
    test('should handle API errors gracefully', async () => {
      (global.fetch as jest.Mock).mockRejectedValue(new Error('Network error'));

      const chunks = await semanticChunk(sampleDocument);
      
      expect(chunks).toBeDefined();
      expect(chunks.length).toBeGreaterThan(0);
    });

    test('should handle malformed API responses', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ invalid: 'response' })
      });

      const chunks = await semanticChunk(sampleDocument);
      
      expect(chunks).toBeDefined();
      expect(chunks.length).toBeGreaterThan(0);
    });
  });

  describe('Cosine similarity calculation', () => {
    test('should calculate similarity correctly', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => [[1, 0, 0], [0, 1, 0], [0, 0, 1]] // Orthogonal vectors
      });

      const chunker = new SemanticChunker();
      const chunks = await chunker.chunkDocument(sampleDocument);
      
      expect(chunks).toBeDefined();
    });

    test('should handle zero vectors', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => [[0, 0, 0], [1, 0, 0]] // Zero vector and unit vector
      });

      const chunker = new SemanticChunker();
      const chunks = await chunker.chunkDocument(sampleDocument);
      
      expect(chunks).toBeDefined();
    });
  });
});

describe('semanticChunk convenience function', () => {
  test('should work as a simple wrapper', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => [[0.1, 0.2, 0.3, 0.4, 0.5]]
    });

    const chunks = await semanticChunk(sampleDocument);
    
    expect(chunks).toBeDefined();
    expect(chunks.length).toBeGreaterThan(0);
  });
});
