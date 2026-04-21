import { SearchChunkByString, searchChunksByString, searchMultipleQueries, findMostSimilarChunks } from '../retrieve/search-chunk-by-string';
import { EmbeddingFormat, SearchOptions } from '../retrieve/search-chunk-by-string';
import { readFileSync } from 'fs';
import { join } from 'path';
import { logger } from '../utils/logger';

/**
 * Example demonstrating search functionality using embedding-results.json
 */

interface QueryConfig {
  id: string;
  text: string;
  category: string;
  topK: number;
  similarityThreshold: number;
}

interface QueryFile {
  version: string;
  created: string;
  description: string;
  queries: QueryConfig[];
  searchOptions: SearchOptions;
}

/**
 * Load queries from JSON file
 */
function loadQueries(): QueryFile {
  const queryFile = join(__dirname, 'example-query.json');
  try {
    const data = JSON.parse(readFileSync(queryFile, 'utf-8'));
    return data;
  } catch (error) {
    logger.error('Failed to load queries from JSON file', error, {
      operation: 'load_queries',
      queryFile
    });
    // Return fallback queries
    return {
      version: '1.0.0',
      created: new Date().toISOString(),
      description: 'Fallback queries',
      queries: [
        {
          id: 'fallback_1',
          text: 'What is artificial intelligence?',
          category: 'basics',
          topK: 3,
          similarityThreshold: 0.1
        }
      ],
      searchOptions: {
        topK: 5,
        similarityThreshold: 0.1,
        embeddingModel: 'Xenova/all-MiniLM-L6-v2',
        embeddingFormat: EmbeddingFormat.NORMALIZED,
        includeMetadata: true
      }
    };
  }
}

async function demonstrateBasicSearch() {
  logger.info('=== Basic Search Demo ===');
  
  const embeddingFile = './src/example/embedding-results.json';
  const queryData = loadQueries();
  
  try {
    // Initialize searcher
    const searcher = new SearchChunkByString(embeddingFile);
    
    // Get statistics
    const stats = searcher.getStats();
    logger.info('Embedding file statistics');
    
    logger.info(`Loaded ${queryData.queries.length} queries from example-query.json`);
    
    
    if (!stats.fileExists || stats.embeddingCount === 0) {
      logger.warn('No embeddings found. Please run the convert-embedding-example first.');
      return;
    }
    
    // Process queries from JSON file
    for (const queryConfig of queryData.queries) {
      logger.info(`Processing query: "${queryConfig.text}"`);
      
      try {
        const results = await searcher.search(queryConfig.text, { 
          topK: queryConfig.topK,
          similarityThreshold: queryConfig.similarityThreshold 
        });
        
        logger.info(`Found ${results.length} results for query: "${queryConfig.text}"`);
        
        if (results.length === 0) {
          logger.info('No results found (try lowering similarity threshold)');
          continue;
        }
        
        results.forEach((result, index) => {
          logger.info(`Result ${index + 1} (Similarity: ${result.similarity.toFixed(4)})`);
          logger.info(`Content: ${result.content.substring(0, 150)}...`);
        });
        
        logger.debug('========================================');
        
      } catch (error) {
        logger.error(`Error searching for "${queryConfig.text}"`, error);
      }
    }
    
  } catch (error) {
    logger.error('Error in basic search demo', error);
  }
}

/**
 * Demonstrate different search options and strategies
 */
async function demonstrateSearchStrategies() {
  logger.info('=== Search Strategies Demo ===');
  
  const embeddingFile = './src/example/embedding-results.json';
  const testQuery = "artificial intelligence and machine learning";
  
  try {
    const searcher = new SearchChunkByString(embeddingFile);
    
    // Test different topK values
    logger.info(`Testing different topK values for query: "${testQuery}"`);
    
    const topKValues = [1, 3, 5, 10];
    
    for (const topK of topKValues) {
      logger.info(`Testing topK ${topK}`);
      
      try {
        const results = await searcher.search(testQuery, { topK });
        
        results.forEach((result, index) => {
          logger.debug(`TopK ${topK} result ${index + 1}`);
        });
      } catch (error) {
        logger.error(`Error with topK ${topK}`);
      }
    }
    
    // Test similarity thresholds
    logger.info('--- Similarity Threshold Filtering ---');
    
    try {
      const allResults = await searcher.search(testQuery, { topK: 10, similarityThreshold: 0 });
      const thresholds = [0.8, 0.6, 0.4, 0.2, 0.1];
      
      for (const threshold of thresholds) {
        const filteredResults = allResults.filter(r => r.similarity >= threshold);
        logger.info(`Threshold ${threshold}: ${filteredResults.length} results`);
        
        if (filteredResults.length > 0) {
          logger.debug(`Best match for threshold ${threshold}`);
        }
      }
    } catch (error) {
      logger.error('Error with threshold testing', error);
    }
    
    // Test different embedding formats
    logger.info('--- Embedding Format Testing ---');
    
    const formats = [EmbeddingFormat.NORMALIZED, EmbeddingFormat.RAW];
    
    for (const format of formats) {
      logger.info(`Testing format: ${format}`);
      
      try {
        const results = await searcher.search(testQuery, { 
          topK: 3, 
          embeddingFormat: format,
          similarityThreshold: 0.1
        });
        
        if (results.length > 0) {
          logger.debug(`Top result for format ${format}`);
        } else {
          logger.info(`No results found for format ${format}`);
        }
      } catch (error) {
        logger.error(`Error with format ${format}`, error);
      }
    }
    
  } catch (error) {
    logger.error('Error in search strategies demo');
  }
}

/**
 * Demonstrate batch searching
 */
async function demonstrateBatchSearch() {
  logger.info('=== Batch Search Demo ===');
  
  const embeddingFile = './src/example/embedding-results.json';
  
  try {
    const queries = [
      "neural networks",
      "computer vision", 
      "natural language processing",
      "deep learning",
      "AI applications"
    ];
    
    logger.info(`Processing ${queries.length} queries in batch`);
    
    // Method 1: Using convenience function
    logger.info('--- Using searchMultipleQueries convenience function ---');
    
    try {
      const batchResults = await searchMultipleQueries(queries, embeddingFile, { topK: 2 });
      
      batchResults.forEach(({ query, results }) => {
        logger.info(`Batch search results for query: "${query}"`);
        if (results.length > 0) {
          results.forEach((result, index) => {
            logger.debug(`Batch search result ${index + 1}`);
          });
        } else {
          logger.info(`No results found for query: "${query}"`);
        }
      });
    } catch (error) {
      logger.error('Error in batch search', error);
    }
    
    // Method 2: Using findMostSimilarChunks
    logger.info('--- Using findMostSimilarChunks function ---');
    
    try {
      const bestMatches = await findMostSimilarChunks(queries, embeddingFile, { topK: 1 });
      
      bestMatches.forEach(({ query, bestMatch }) => {
        logger.info(`Best match for query: "${query}"`);
      });
    } catch (error) {
      logger.error('Error finding best matches', error);
    }
    
  } catch (error) {
    logger.error('Error in batch search demo', error);
  }
}

/**
 * Demonstrate advanced search features
 */
async function demonstrateAdvancedSearch() {
  logger.info('=== Advanced Search Demo ===');
  
  const embeddingFile = './src/example/embedding-results.json';
  
  try {
    const searcher = new SearchChunkByString(embeddingFile);
    
    // Show all available chunks
    logger.info('--- Available Chunks Preview ---');
    
    try {
      const allChunks = searcher.getAllChunks();
      logger.info(`Total chunks: ${allChunks.length}`);
      
      allChunks.slice(0, 3).forEach((chunk, index) => {
        logger.debug(`Chunk ${index + 1} preview`);
      });
      
      if (allChunks.length > 3) {
        logger.info(`... and ${allChunks.length - 3} more chunks`);
      }
    } catch (error) {
      logger.error('Error getting chunks', error);
    }
    
    // Test with custom search options
    logger.info('--- Custom Search Options ---');
    
    const customOptions: SearchOptions = {
      topK: 2,
      similarityThreshold: 0.3,
      embeddingModel: 'Xenova/all-MiniLM-L6-v2',
      embeddingFormat: EmbeddingFormat.NORMALIZED,
      includeMetadata: true
    };
    
    const advancedQueries = [
      "What are the ethical considerations in AI development?",
      "How does deep learning differ from traditional machine learning?",
      "What are the applications of computer vision?"
    ];
    
    for (const query of advancedQueries) {
      logger.info(`Processing advanced query: "${query}"`);
      
      try {
        const results = await searcher.search(query, customOptions);
        
        results.forEach((result, index) => {
          logger.debug(`Advanced search result ${index + 1} (similarity: ${result.similarity.toFixed(4)})`);
          logger.debug(`Result content: ${result.content}`);
        });
      } catch (error) {
        logger.error(`Error with advanced query "${query}"`, error);
      }
      
      logger.debug('========================================');
    }
    
  } catch (error) {
    logger.error('Error in advanced search demo', error);
  }
}

// Export for use in other modules
export {
  demonstrateBasicSearch,
  demonstrateSearchStrategies,
  demonstrateBatchSearch,
  demonstrateAdvancedSearch
};

// Export the main function to run all demos
export async function runAllSearchDemos() {
  try {
    // await demonstrateBasicSearch();
    // await demonstrateSearchStrategies();
    // await demonstrateBatchSearch();
    await demonstrateAdvancedSearch();
  } catch (error) {
    logger.error('Error running search demos', error);
  }
}

// Run if this file is executed directly
if (require.main === module) {
  runAllSearchDemos().catch(error => logger.error('Unhandled error in search demos', error));
}
