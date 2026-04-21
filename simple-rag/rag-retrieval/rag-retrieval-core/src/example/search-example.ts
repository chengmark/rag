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
  logger.info('=== Basic Search Demo ===', {
    operation: 'demo_basic_search'
  });
  
  const embeddingFile = './src/example/embedding-results.json';
  const queryData = loadQueries();
  
  try {
    // Initialize searcher
    const searcher = new SearchChunkByString(embeddingFile);
    
    // Get statistics
    const stats = searcher.getStats();
    logger.info('Embedding file statistics', {
      operation: 'file_stats',
      fileExists: stats.fileExists,
      embeddingCount: stats.embeddingCount,
      dimensions: stats.dimensions,
      model: stats.model,
      format: stats.format
    });
    
    logger.info(`Loaded ${queryData.queries.length} queries from example-query.json`, {
      operation: 'load_queries',
      queryCount: queryData.queries.length
    });
    
    if (!stats.fileExists || stats.embeddingCount === 0) {
      logger.warn('No embeddings found. Please run the convert-embedding-example first.', {
        operation: 'demo_basic_search',
        embeddingFile
      });
      return;
    }
    
    // Process queries from JSON file
    for (const queryConfig of queryData.queries) {
      logger.info(`Processing query: "${queryConfig.text}"`, {
        operation: 'process_query',
        queryId: queryConfig.id,
        category: queryConfig.category,
        topK: queryConfig.topK,
        threshold: queryConfig.similarityThreshold
      });
      
      try {
        const results = await searcher.search(queryConfig.text, { 
          topK: queryConfig.topK,
          similarityThreshold: queryConfig.similarityThreshold 
        });
        
        logger.info(`Found ${results.length} results for query: "${queryConfig.text}"`, {
          operation: 'search_results',
          queryId: queryConfig.id,
          resultCount: results.length
        });
        
        if (results.length === 0) {
          logger.info('No results found (try lowering similarity threshold)', {
            operation: 'search_results',
            queryId: queryConfig.id,
            threshold: queryConfig.similarityThreshold
          });
          continue;
        }
        
        results.forEach((result, index) => {
          logger.info(`Result ${index + 1} (Similarity: ${result.similarity.toFixed(4)})`, {
            operation: 'search_result',
            queryId: queryConfig.id,
            index: index + 1,
            similarity: result.similarity,
            content: result.content.substring(0, 150),
            chunkIndex: result.chunkIndex,
            source: result.source,
            metadata: Object.keys(result.metadata).length > 0 
              ? JSON.stringify(result.metadata, null, 6).substring(0, 100)
              : undefined
          });
          logger.info(`Content: ${result.content.substring(0, 150)}...`);
        });
        
        logger.debug('Separator line', {
          operation: 'separator'
        });
        
      } catch (error) {
        logger.error(`Error searching for "${queryConfig.text}"`, error, {
          operation: 'search_error',
          queryId: queryConfig.id,
          queryText: queryConfig.text
        });
      }
    }
    
  } catch (error) {
    logger.error('Error in basic search demo', error, {
      operation: 'demo_basic_search'
    });
  }
}

/**
 * Demonstrate different search options and strategies
 */
async function demonstrateSearchStrategies() {
  logger.info('=== Search Strategies Demo ===', {
    operation: 'demo_search_strategies'
  });
  
  const embeddingFile = './src/example/embedding-results.json';
  const testQuery = "artificial intelligence and machine learning";
  
  try {
    const searcher = new SearchChunkByString(embeddingFile);
    
    // Test different topK values
    logger.info(`Testing different topK values for query: "${testQuery}"`, {
      operation: 'test_topk_values',
      query: testQuery
    });
    
    const topKValues = [1, 3, 5, 10];
    
    for (const topK of topKValues) {
      logger.info(`Testing topK ${topK}`, {
        operation: 'test_topk',
        topK,
        query: testQuery
      });
      
      try {
        const results = await searcher.search(testQuery, { topK });
        
        results.forEach((result, index) => {
          logger.debug(`TopK ${topK} result ${index + 1}`, {
            operation: 'test_topk_result',
            topK,
            index: index + 1,
            similarity: result.similarity,
            content: result.content.substring(0, 100)
          });
        });
      } catch (error) {
        logger.error(`Error with topK ${topK}`, error, {
          operation: 'test_topk',
          topK,
          query: testQuery
        });
      }
    }
    
    // Test similarity thresholds
    logger.info('--- Similarity Threshold Filtering ---', {
      operation: 'test_thresholds',
      query: testQuery
    });
    
    try {
      const allResults = await searcher.search(testQuery, { topK: 10, similarityThreshold: 0 });
      const thresholds = [0.8, 0.6, 0.4, 0.2, 0.1];
      
      for (const threshold of thresholds) {
        const filteredResults = allResults.filter(r => r.similarity >= threshold);
        logger.info(`Threshold ${threshold}: ${filteredResults.length} results`, {
          operation: 'test_threshold',
          threshold,
          resultCount: filteredResults.length
        });
        
        if (filteredResults.length > 0) {
          logger.debug(`Best match for threshold ${threshold}`, {
            operation: 'test_threshold',
            threshold,
            similarity: filteredResults[0].similarity,
            content: filteredResults[0].content.substring(0, 80)
          });
        }
      }
    } catch (error) {
      logger.error('Error with threshold testing', error, {
        operation: 'test_thresholds',
        query: testQuery
      });
    }
    
    // Test different embedding formats
    logger.info('--- Embedding Format Testing ---', {
      operation: 'test_formats',
      query: testQuery
    });
    
    const formats = [EmbeddingFormat.NORMALIZED, EmbeddingFormat.RAW];
    
    for (const format of formats) {
      logger.info(`Testing format: ${format}`, {
        operation: 'test_format',
        format,
        query: testQuery
      });
      
      try {
        const results = await searcher.search(testQuery, { 
          topK: 3, 
          embeddingFormat: format,
          similarityThreshold: 0.1
        });
        
        if (results.length > 0) {
          logger.debug(`Top result for format ${format}`, {
            operation: 'test_format',
            format,
            similarity: results[0].similarity,
            content: results[0].content.substring(0, 80)
          });
        } else {
          logger.info(`No results found for format ${format}`, {
            operation: 'test_format',
            format
          });
        }
      } catch (error) {
        logger.error(`Error with format ${format}`, error, {
          operation: 'test_format',
          format,
          query: testQuery
        });
      }
    }
    
  } catch (error) {
    logger.error('Error in search strategies demo', error, {
      operation: 'demo_search_strategies'
    });
  }
}

/**
 * Demonstrate batch searching
 */
async function demonstrateBatchSearch() {
  logger.info('=== Batch Search Demo ===', {
    operation: 'demo_batch_search'
  });
  
  const embeddingFile = './src/example/embedding-results.json';
  
  try {
    const queries = [
      "neural networks",
      "computer vision", 
      "natural language processing",
      "deep learning",
      "AI applications"
    ];
    
    logger.info(`Processing ${queries.length} queries in batch`, {
      operation: 'batch_search',
      queryCount: queries.length
    });
    
    // Method 1: Using convenience function
    logger.info('--- Using searchMultipleQueries convenience function ---', {
      operation: 'batch_search_method1'
    });
    
    try {
      const batchResults = await searchMultipleQueries(queries, embeddingFile, { topK: 2 });
      
      batchResults.forEach(({ query, results }) => {
        logger.info(`Batch search results for query: "${query}"`, {
          operation: 'batch_search_results',
          query,
          resultCount: results.length
        });
        if (results.length > 0) {
          results.forEach((result, index) => {
            logger.debug(`Batch search result ${index + 1}`, {
              operation: 'batch_search_result',
              query,
              index: index + 1,
              similarity: result.similarity,
              content: result.content.substring(0, 80)
            });
          });
        } else {
          logger.info(`No results found for query: "${query}"`, {
            operation: 'batch_search_results',
            query,
            resultCount: 0
          });
        }
      });
    } catch (error) {
      logger.error('Error in batch search', error, {
        operation: 'batch_search_method1'
      });
    }
    
    // Method 2: Using findMostSimilarChunks
    logger.info('--- Using findMostSimilarChunks function ---', {
      operation: 'batch_search_method2'
    });
    
    try {
      const bestMatches = await findMostSimilarChunks(queries, embeddingFile, { topK: 1 });
      
      bestMatches.forEach(({ query, bestMatch }) => {
        logger.info(`Best match for query: "${query}"`, {
          operation: 'best_match',
          query,
          similarity: bestMatch.similarity,
          content: bestMatch.content.substring(0, 100)
        });
      });
    } catch (error) {
      logger.error('Error finding best matches', error, {
        operation: 'batch_search_method2'
      });
    }
    
  } catch (error) {
    logger.error('Error in batch search demo', error, {
      operation: 'demo_batch_search'
    });
  }
}

/**
 * Demonstrate advanced search features
 */
async function demonstrateAdvancedSearch() {
  logger.info('=== Advanced Search Demo ===', {
    operation: 'demo_advanced_search'
  });
  
  const embeddingFile = './src/example/embedding-results.json';
  
  try {
    const searcher = new SearchChunkByString(embeddingFile);
    
    // Show all available chunks
    logger.info('--- Available Chunks Preview ---', {
      operation: 'preview_chunks'
    });
    
    try {
      const allChunks = searcher.getAllChunks();
      logger.info(`Total chunks: ${allChunks.length}`, {
        operation: 'preview_chunks',
        totalChunks: allChunks.length
      });
      
      allChunks.slice(0, 3).forEach((chunk, index) => {
        logger.debug(`Chunk ${index + 1} preview`, {
          operation: 'preview_chunk',
          index: index + 1,
          content: chunk.content.substring(0, 150),
          metadata: JSON.stringify(chunk.metadata, null, 2).substring(0, 200)
        });
      });
      
      if (allChunks.length > 3) {
        logger.info(`... and ${allChunks.length - 3} more chunks`, {
          operation: 'preview_chunks',
          remainingChunks: allChunks.length - 3
        });
      }
    } catch (error) {
      logger.error('Error getting chunks', error, {
        operation: 'preview_chunks'
      });
    }
    
    // Test with custom search options
    logger.info('--- Custom Search Options ---', {
      operation: 'custom_options'
    });
    
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
      logger.info(`Processing advanced query: "${query}"`, {
        operation: 'advanced_query',
        query,
        options: customOptions
      });
      
      try {
        const results = await searcher.search(query, customOptions);
        
        results.forEach((result, index) => {
          logger.debug(`Advanced search result ${index + 1}`, {
            operation: 'advanced_result',
            query,
            index: index + 1,
            similarity: result.similarity,
            content: result.content.substring(0, 120),
            chunkIndex: result.chunkIndex,
            source: result.source
          });
          
          if (Object.keys(result.metadata).length > 0) {
            logger.debug(`Metadata sample for result ${index + 1}`, {
              operation: 'advanced_result',
              query,
              metadataSample: Object.entries(result.metadata).slice(0, 3)
            });
          }
        });
      } catch (error) {
        logger.error(`Error with advanced query "${query}"`, error, {
          operation: 'advanced_query',
          query
        });
      }
      
      logger.debug('Separator line', {
        operation: 'separator'
      });
    }
    
  } catch (error) {
    logger.error('Error in advanced search demo', error, {
      operation: 'demo_advanced_search'
    });
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
    await demonstrateBasicSearch();
    // await demonstrateSearchStrategies();
    // await demonstrateBatchSearch();
    // await demonstrateAdvancedSearch();
  } catch (error) {
    logger.error('Error running search demos', error, {
      operation: 'run_all_demos'
    });
  }
}

// Run if this file is executed directly
if (require.main === module) {
  runAllSearchDemos().catch(error => logger.error('Unhandled error in search demos', error, {
    operation: 'main'
  }));
}
