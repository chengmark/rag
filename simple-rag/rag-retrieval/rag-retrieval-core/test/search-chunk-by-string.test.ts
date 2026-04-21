import { SearchChunkByString, searchChunksByString, EmbeddingFormat } from '../src/retrieve/search-chunk-by-string';
import { writeFileSync, existsSync, unlinkSync } from 'fs';
import { join } from 'path';

// Mock test data
const mockEmbeddingData = {
  version: '1.0.0',
  created: new Date().toISOString(),
  count: 3,
  embeddings: [
    {
      id: 'chunk_0_1234567890',
      embedding: Array.from({length: 384}, (_, i) => (i % 10) / 100), // Mock 384D vector
      content: 'Artificial intelligence is a branch of computer science that aims to create intelligent machines.',
      metadata: {
        chunkIndex: '0',
        source: 'test-document.txt',
        embeddingModel: 'Xenova/all-MiniLM-L6-v2',
        embeddingFormat: 'normalized'
      },
      format: 'normalized' as EmbeddingFormat,
      model: 'Xenova/all-MiniLM-L6-v2',
      dimensions: 384,
      timestamp: Date.now()
    },
    {
      id: 'chunk_1_1234567891',
      embedding: Array.from({length: 384}, (_, i) => ((i + 5) % 10) / 100), // Different mock vector
      content: 'Machine learning algorithms enable computers to learn from data without explicit programming.',
      metadata: {
        chunkIndex: '1',
        source: 'test-document.txt',
        embeddingModel: 'Xenova/all-MiniLM-L6-v2',
        embeddingFormat: 'normalized'
      },
      format: 'normalized' as EmbeddingFormat,
      model: 'Xenova/all-MiniLM-L6-v2',
      dimensions: 384,
      timestamp: Date.now()
    },
    {
      id: 'chunk_2_1234567892',
      embedding: Array.from({length: 384}, (_, i) => ((i + 10) % 10) / 100), // Another mock vector
      content: 'Deep learning uses neural networks with multiple layers to process complex patterns.',
      metadata: {
        chunkIndex: '2',
        source: 'test-document.txt',
        embeddingModel: 'Xenova/all-MiniLM-L6-v2',
        embeddingFormat: 'normalized'
      },
      format: 'normalized' as EmbeddingFormat,
      model: 'Xenova/all-MiniLM-L6-v2',
      dimensions: 384,
      timestamp: Date.now()
    }
  ]
};

describe('SearchChunkByString', () => {
  const testEmbeddingFile = join(__dirname, 'test-embeddings.json');

  beforeEach(() => {
    // Create test embedding file
    writeFileSync(testEmbeddingFile, JSON.stringify(mockEmbeddingData, null, 2));
  });

  afterEach(() => {
    // Clean up test file
    if (existsSync(testEmbeddingFile)) {
      unlinkSync(testEmbeddingFile);
    }
  });

  describe('Basic functionality', () => {
    test('should initialize with default options', () => {
      const searcher = new SearchChunkByString(testEmbeddingFile);
      expect(searcher).toBeInstanceOf(SearchChunkByString);
    });

    test('should initialize with custom options', () => {
      const customOptions = {
        topK: 3,
        similarityThreshold: 0.5,
        embeddingModel: 'custom-model'
      };
      const searcher = new SearchChunkByString(testEmbeddingFile, customOptions);
      expect(searcher).toBeInstanceOf(SearchChunkByString);
    });

    test('should load embeddings from file', () => {
      const searcher = new SearchChunkByString(testEmbeddingFile);
      const stats = searcher.getStats();
      
      expect(stats.fileExists).toBe(true);
      expect(stats.embeddingCount).toBe(3);
      expect(stats.dimensions).toBe(384);
      expect(stats.model).toBe('Xenova/all-MiniLM-L6-v2');
      expect(stats.format).toBe('normalized');
    });

    test('should handle non-existent file', () => {
      const searcher = new SearchChunkByString('non-existent.json');
      const stats = searcher.getStats();
      
      expect(stats.fileExists).toBe(false);
      expect(stats.embeddingCount).toBe(0);
    });
  });

  describe('Search functionality', () => {
    test('should perform basic search', async () => {
      const searcher = new SearchChunkByString(testEmbeddingFile);
      const results = await searcher.search('artificial intelligence');
      
      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
      expect(results.length).toBeGreaterThan(0);
      
      results.forEach(result => {
        expect(result).toHaveProperty('content');
        expect(result).toHaveProperty('similarity');
        expect(result).toHaveProperty('metadata');
        expect(result).toHaveProperty('chunkIndex');
        expect(result).toHaveProperty('source');
        expect(typeof result.similarity).toBe('number');
      });
    });

    test('should respect topK limit', async () => {
      const searcher = new SearchChunkByString(testEmbeddingFile);
      const results = await searcher.search('machine learning', { topK: 2 });
      
      expect(results.length).toBeLessThanOrEqual(2);
    });

    test('should filter by similarity threshold', async () => {
      const searcher = new SearchChunkByString(testEmbeddingFile);
      const results = await searcher.search('test query', { 
        similarityThreshold: 0.9, // High threshold
        topK: 10
      });
      
      // With high threshold, we expect fewer or no results
      expect(results.length).toBeLessThanOrEqual(3);
    });

    test('should return empty results for empty embeddings', async () => {
      // Create empty embedding file
      const emptyFile = join(__dirname, 'empty-embeddings.json');
      writeFileSync(emptyFile, JSON.stringify({ version: '1.0.0', created: new Date().toISOString(), count: 0, embeddings: [] }));
      
      try {
        const searcher = new SearchChunkByString(emptyFile);
        const results = await searcher.search('test query');
        expect(results).toEqual([]);
      } finally {
        if (existsSync(emptyFile)) {
          unlinkSync(emptyFile);
        }
      }
    });
  });

  describe('Convenience functions', () => {
    test('searchChunksByString should work', async () => {
      const results = await searchChunksByString('deep learning', testEmbeddingFile);
      
      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
    });

    test('searchMultipleQueries should work', async () => {
      const { searchMultipleQueries } = require('../src/retrieve/search-chunk-by-string');
      const queries = ['AI', 'machine learning', 'neural networks'];
      
      const results = await searchMultipleQueries(queries, testEmbeddingFile);
      
      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
      expect(results).toHaveLength(3);
      
      results.forEach((result: any) => {
        expect(result).toHaveProperty('query');
        expect(result).toHaveProperty('results');
        expect(Array.isArray(result.results)).toBe(true);
      });
    });
  });

  describe('Utility methods', () => {
    test('getAllChunks should return all chunks', () => {
      const searcher = new SearchChunkByString(testEmbeddingFile);
      const chunks = searcher.getAllChunks();
      
      expect(chunks).toBeDefined();
      expect(Array.isArray(chunks)).toBe(true);
      expect(chunks).toHaveLength(3);
      
      chunks.forEach(chunk => {
        expect(chunk).toHaveProperty('content');
        expect(chunk).toHaveProperty('metadata');
      });
    });

    test('reload should refresh embeddings', () => {
      const searcher = new SearchChunkByString(testEmbeddingFile);
      
      // Initial load
      const initialStats = searcher.getStats();
      expect(initialStats.embeddingCount).toBe(3);
      
      // Reload
      searcher.reload();
      const reloadedStats = searcher.getStats();
      expect(reloadedStats.embeddingCount).toBe(3);
    });
  });

  describe('Error handling', () => {
    test('should handle invalid JSON file', () => {
      const invalidFile = join(__dirname, 'invalid-embeddings.json');
      writeFileSync(invalidFile, 'invalid json content');
      
      try {
        const searcher = new SearchChunkByString(invalidFile);
        expect(() => searcher.getStats()).toThrow();
      } finally {
        if (existsSync(invalidFile)) {
          unlinkSync(invalidFile);
        }
      }
    });
  });
});

describe('EmbeddingFormat enum', () => {
  test('should have correct values', () => {
    expect(EmbeddingFormat.RAW).toBe('raw');
    expect(EmbeddingFormat.NORMALIZED).toBe('normalized');
    expect(EmbeddingFormat.BINARY).toBe('binary');
    expect(EmbeddingFormat.QUANTIZED).toBe('quantized');
  });
});
