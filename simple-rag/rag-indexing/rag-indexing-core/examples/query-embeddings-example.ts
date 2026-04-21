import { generateEmbedding, searchEmbeddings } from '../src/index';
import { EmbeddingFormat } from '../src/embedding/convert-embedding';
import { logger } from '../src/utils/logger';

/**
 * Example demonstrating how to query saved embeddings for retrieval
 */

async function demonstrateEmbeddingQuery() {
  );

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

    );
    );

    for (let i = 0; i < queries.length; i++) {
      const query = queries[i];
      );
      
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

        );
        
        results.forEach((result, index) => {
          );
        });

        );

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
      );
    }
  }
}

/**
 * Demonstrate batch querying for efficiency
 */
async function demonstrateBatchQuerying() {
  );

  const embeddingFile = './examples/embedding-results.json';
  
  try {
    const batchQueries = [
      { id: 'ml_basics', text: 'machine learning fundamentals' },
      { id: 'deep_learning', text: 'deep learning architectures' },
      { id: 'nlp_concepts', text: 'natural language processing concepts' },
      { id: 'computer_vision', text: 'computer vision applications' },
      { id: 'ai_ethics', text: 'AI ethics and responsibility' }
    ];

    );

    // Generate all query embeddings first
    );
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
      );
      
      const results = await searchEmbeddings(queryData.embedding, 2, embeddingFile);
      
      if (results.length > 0) {
        );
      } else {
        );
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
  );

  const embeddingFile = './examples/embedding-results.json';
  const testQuery = "artificial intelligence and machine learning fundamentals";

  try {
    );

    // Generate query embedding once
    const queryEmbedding = await generateEmbedding(testQuery, 'Xenova/all-MiniLM-L6-v2', {
      targetFormat: EmbeddingFormat.NORMALIZED
    });

    // Test different topK values
    const topKValues = [1, 3, 5, 10];
    
    for (const topK of topKValues) {
      );
      
      const results = await searchEmbeddings(
        (queryEmbedding as any).vector, 
        topK, 
        embeddingFile
      );

      results.forEach((result, index) => {
        );
      });
    }

    // Similarity threshold filtering
    );
    const allResults = await searchEmbeddings((queryEmbedding as any).vector, 10, embeddingFile);
    
    const thresholds = [0.8, 0.6, 0.4, 0.2];
    
    for (const threshold of thresholds) {
      const filteredResults = allResults.filter(r => r.similarity >= threshold);
      );
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

