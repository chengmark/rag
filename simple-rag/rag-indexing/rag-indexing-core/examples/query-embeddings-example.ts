import { generateEmbedding, searchEmbeddings } from '../src/index';
import { EmbeddingFormat } from '../src/embedding/convert-embedding';
import { logger } from '../src/utils/logger';

/**
 * Example demonstrating how to query saved embeddings for retrieval
 */

async function demonstrateEmbeddingQuery() {
  logger.info('=== Embedding Query Demo ===', {
    operation: 'demo_embedding_query'
  });

  const embeddingFile = './examples/embedding-results.json';
  
  try {
    // Sample queries for different topics
    const queries = [
      "What is artificial intelligence and machine learning?",
      "How do neural networks work in deep learning?", 
      "What are the applications of computer vision?",
      "Explain natural language processing and transformers",
      "What are the ethical considerations in AI development?"
    ];

    logger.info(`Loading embeddings from: ${embeddingFile}`, {
      operation: 'load_embeddings',
      embeddingFile
    });
    logger.info(`Processing ${queries.length} sample queries`, {
      operation: 'process_queries',
      queryCount: queries.length
    });

    for (let i = 0; i < queries.length; i++) {
      const query = queries[i];
      logger.info(`--- Query ${i + 1} ---`, {
        operation: 'process_query',
        index: i + 1,
        query
      });
      
      try {
        // Generate embedding for the query
        const queryEmbedding = await generateEmbedding(query, 'Xenova/all-MiniLM-L6-v2', {
          targetFormat: EmbeddingFormat.NORMALIZED
        });

        // Search for similar chunks
        const results = await searchEmbeddings(
          (queryEmbedding as any).vector, 
          3, 
          embeddingFile
        );

        logger.info(`Found ${results.length} relevant chunks`, {
          operation: 'search_results',
          index: i + 1,
          resultCount: results.length
        });
        
        results.forEach((result, index) => {
          logger.info(`Result ${index + 1} (Similarity: ${result.similarity.toFixed(4)})`, {
            operation: 'search_result',
            queryIndex: i + 1,
            resultIndex: index + 1,
            similarity: result.similarity,
            content: result.content.substring(0, 200),
            chunkIndex: result.embedding.metadata.chunkIndex,
            source: result.embedding.metadata.source
          });
        });

        logger.debug('Separator line', {
          operation: 'separator'
        });

      } catch (error) {
        logger.error(`Error processing query "${query}"`, error, {
          operation: 'process_query_error',
          query
        });
      }
    }

  } catch (error) {
    logger.error('Error in embedding query demo', error, {
      operation: 'demo_embedding_query'
    });
    
    if (error instanceof Error) {
      logger.info('Troubleshooting tips for embedding query', {
        operation: 'troubleshooting',
        tips: [
          'Ensure embedding-results.json exists from running convert-embedding-example.ts',
          'Check if @huggingface/transformers is installed',
          'Verify the embedding file path is correct',
          'Make sure you have enough memory for the model'
        ]
      });
    }
  }
}

/**
 * Demonstrate batch querying for efficiency
 */
async function demonstrateBatchQuerying() {
  logger.info('=== Batch Querying Demo ===', {
    operation: 'demo_batch_querying'
  });

  const embeddingFile = './examples/embedding-results.json';
  
  try {
    const batchQueries = [
      { id: 'ml_basics', text: 'machine learning fundamentals' },
      { id: 'deep_learning', text: 'deep learning architectures' },
      { id: 'nlp_concepts', text: 'natural language processing concepts' },
      { id: 'computer_vision', text: 'computer vision applications' },
      { id: 'ai_ethics', text: 'AI ethics and responsibility' }
    ];

    logger.info(`Processing ${batchQueries.length} queries in batch`, {
      operation: 'batch_processing',
      queryCount: batchQueries.length
    });

    // Generate all query embeddings first
    logger.info('Generating query embeddings...', {
      operation: 'generate_embeddings',
      queryCount: batchQueries.length
    });
    const queryEmbeddings = await Promise.all(
      batchQueries.map(async (query) => {
        const embedding = await generateEmbedding(query.text, 'Xenova/all-MiniLM-L6-v2', {
          targetFormat: EmbeddingFormat.NORMALIZED
        });
        return { ...query, embedding: (embedding as any).vector };
      })
    );

    // Search for each query
    for (const queryData of queryEmbeddings) {
      logger.info(`Batch query: ${queryData.text} (${queryData.id})`, {
        operation: 'batch_query',
        id: queryData.id,
        query: queryData.text
      });
      
      const results = await searchEmbeddings(queryData.embedding, 2, embeddingFile);
      
      if (results.length > 0) {
        logger.info(`Top result for batch query: ${queryData.text}`, {
          operation: 'batch_result',
          id: queryData.id,
          similarity: results[0].similarity,
          content: results[0].content.substring(0, 150)
        });
      } else {
        logger.info(`No results found for batch query: ${queryData.text}`, {
          operation: 'batch_result',
          id: queryData.id,
          resultCount: 0
        });
      }
    }

  } catch (error) {
    logger.error('Error in batch querying demo', error, {
      operation: 'demo_batch_querying'
    });
  }
}

/**
 * Demonstrate different search strategies
 */
async function demonstrateSearchStrategies() {
  logger.info('=== Search Strategies Demo ===', {
    operation: 'demo_search_strategies'
  });

  const embeddingFile = './examples/embedding-results.json';
  const testQuery = "artificial intelligence and machine learning fundamentals";

  try {
    logger.info(`Test query: "${testQuery}"`, {
      operation: 'test_query',
      query: testQuery
    });

    // Generate query embedding once
    const queryEmbedding = await generateEmbedding(testQuery, 'Xenova/all-MiniLM-L6-v2', {
      targetFormat: EmbeddingFormat.NORMALIZED
    });

    // Test different topK values
    const topKValues = [1, 3, 5, 10];
    
    for (const topK of topKValues) {
      logger.info(`Testing topK ${topK}`, {
        operation: 'test_topk',
        topK,
        query: testQuery
      });
      
      const results = await searchEmbeddings(
        (queryEmbedding as any).vector, 
        topK, 
        embeddingFile
      );

      results.forEach((result, index) => {
        logger.debug(`TopK ${topK} result ${index + 1}`, {
          operation: 'test_topk_result',
          topK,
          index: index + 1,
          similarity: result.similarity,
          content: result.content.substring(0, 100)
        });
      });
    }

    // Similarity threshold filtering
    logger.info('--- Similarity Threshold Filtering ---', {
      operation: 'test_thresholds',
      query: testQuery
    });
    const allResults = await searchEmbeddings((queryEmbedding as any).vector, 10, embeddingFile);
    
    const thresholds = [0.8, 0.6, 0.4, 0.2];
    
    for (const threshold of thresholds) {
      const filteredResults = allResults.filter(r => r.similarity >= threshold);
      logger.info(`Threshold ${threshold}: ${filteredResults.length} results`, {
        operation: 'test_threshold',
        threshold,
        resultCount: filteredResults.length
      });
    }

  } catch (error) {
    logger.error('Error in search strategies demo', error, {
      operation: 'demo_search_strategies'
    });
  }
}

// Export for use in other modules
export { 
  demonstrateEmbeddingQuery,
  demonstrateBatchQuerying,
  demonstrateSearchStrategies
};

// Run if this file is executed directly
if (require.main === module) {
  (async () => {
    await demonstrateEmbeddingQuery();
    await demonstrateBatchQuerying();
    await demonstrateSearchStrategies();
  })().catch(error => logger.error('Unhandled error in embedding query demo', error, {
    operation: 'main'
  }));
}
