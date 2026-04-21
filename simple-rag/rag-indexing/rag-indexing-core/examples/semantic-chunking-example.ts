import { Document, loadTxt } from '../src/index';
import { SemanticChunker, SemanticChunkingOptions, semanticChunk } from '../src/chunking/semantic-chunking';
import { logger } from '../src/utils/logger';

/**
 * Example demonstrating semantic chunking with local Hugging Face transformers
 */

async function demonstrateSemanticChunking() {
  // Sample document for chunking
  const sampleDocument: Document = {
    content: `Artificial intelligence (AI) is transforming the way we work and live. Machine learning, a subset of AI, enables computers to learn from data without explicit programming. Deep learning, which uses neural networks with multiple layers, has revolutionized fields like computer vision and natural language processing.

Natural language processing (NLP) allows machines to understand and generate human language. Modern NLP models like GPT and BERT have achieved remarkable performance on various language tasks. These models are trained on vast amounts of text data using techniques like transformer architectures.

Computer vision is another area where AI has made significant prfogress. Convolutional neural networks (CNNs) can now identify objects in images with high accuracy. Applications range from autonomous vehicles to medical imaging diagnosis.

The future of AI looks promising, with ongoing research in areas like reinforcement learning, generative models, and explainable AI. However, ethical considerations and responsible AI development remain crucial as these technologies become more prevalent in society.`,
    metadata: {
      title: 'Introduction to Artificial Intelligence',
      author: 'AI Expert',
      category: 'technology'
    }
  };

  // Configure semantic chunking options
  const options: SemanticChunkingOptions = {
    maxChunkSize: 300,        // Maximum characters per chunk
    minChunkSize: 50,         // Minimum characters per chunk
    similarityThreshold: 0.3,  // Similarity threshold for splitting
    embeddingModel: 'Xenova/all-MiniLM-L6-v2', // Local model
    overlapSize: 20           // Overlap between chunks
  };

  try {
    );
    const chunker = new SemanticChunker(options);

    // Get model information
    const modelInfo = await chunker.getModelInfo();
    );

    );
    const startTime = Date.now();
    
    // Perform semantic chunking
    const chunks = await chunker.chunkDocument(sampleDocument);
    
    const endTime = Date.now();
    );

    // Display results
    );
    
    chunks.forEach((chunk, index) => {
      );
    });

    // Show cache statistics
    const cacheStats = chunker.getCacheStats();
    );

    // Example with different model
    );
    const bgeOptions: SemanticChunkingOptions = {
      ...options,
      embeddingModel: 'Xenova/bge-m3'
    };
    
    const bgeChunker = new SemanticChunker(bgeOptions);
    const bgeChunks = await bgeChunker.chunkDocument(sampleDocument);
    
    );

    // Clean up
    chunker.clearCache(true); // Clear cache and unload model
    bgeChunker.clearCache(true);

  } catch (error) {
    logger.error('Error during semantic chunking', error, {
      operation: 'semantic_chunking_error'
    });
    
    if (error instanceof Error) {
      logger.error('Error details', error, {
        operation: 'error_details',
        errorMessage: error.message
      });
      
      // Helpful troubleshooting information
      );
    }
  }
}

// Example of using the convenience function
async function demonstrateConvenienceFunction() {
  const document: Document = {
    content: `The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet at least once. It's commonly used for testing fonts and keyboards.`,
    metadata: { source: 'example' }
  };

  try {
    );
    const chunks = await semanticChunk(document, {
      maxChunkSize: 100,
      similarityThreshold: 0.4
    });
    
    );
    chunks.forEach((chunk: Document, i: number) => {
      );
    });
  } catch (error) {
    logger.error('Error with convenience function', error, {
      operation: 'convenience_function_error'
    });
  }
}

// Export for use in other modules
export { demonstrateSemanticChunking, demonstrateConvenienceFunction };

/**
 * Demonstrate loading text files with the txt loader before chunking
 */
async function demonstrateTxtLoaderWithChunking() {
  );
  
  try {
    // Load the example document using txt loader
    const documentPath = './examples/example-document.txt';
    );
    
    const document = await loadTxt(documentPath, {
      metadata: {
        category: 'technology',
        author: 'AI Expert'
      }
    });
    
    );
    
    // Display some metadata
    );
    
    // Now perform semantic chunking on the loaded document
    );
    const chunker = new SemanticChunker({
      maxChunkSize: 300,
      minChunkSize: 50,
      similarityThreshold: 0.3,
      embeddingModel: 'Xenova/all-MiniLM-L6-v2'
    });
    
    const chunks = await chunker.chunkDocument(document);
    );
    
    // Show first few chunks
    chunks.slice(0, 3).forEach((chunk, index) => {
      );
    });
    
    chunker.clearCache(true);
    
  } catch (error) {
    logger.error('Error with txt loader demonstration', error, {
      operation: 'txt_loader_error'
    });
    );
  }
}

// Run if this file is executed directly
if (require.main === module) {
  (async () => {
    await demonstrateSemanticChunking();
    await demonstrateTxtLoaderWithChunking();
    await demonstrateConvenienceFunction();
  })().catch(error => logger.error('Unhandled error in semantic chunking demo', error, {
    operation: 'main'
  }));
}

