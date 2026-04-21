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
    logger.info('Initializing SemanticChunker...', {
      operation: 'initialize_chunker',
      model: options.embeddingModel
    });
    const chunker = new SemanticChunker(options);

    // Get model information
    const modelInfo = await chunker.getModelInfo();
    logger.info('Model information', {
      operation: 'model_info',
      model: modelInfo.model,
      loaded: modelInfo.loaded,
      cacheSize: modelInfo.cacheSize
    });

    logger.info('Starting semantic chunking...', {
      operation: 'start_chunking',
      documentLength: sampleDocument.content.length
    });
    const startTime = Date.now();
    
    // Perform semantic chunking
    const chunks = await chunker.chunkDocument(sampleDocument);
    
    const endTime = Date.now();
    logger.info(`Chunking completed in ${endTime - startTime}ms`, {
      operation: 'chunking_completed',
      duration: endTime - startTime,
      chunkCount: chunks.length
    });

    // Display results
    logger.info(`Generated ${chunks.length} semantic chunks`, {
      operation: 'chunking_results',
      chunkCount: chunks.length
    });
    
    chunks.forEach((chunk, index) => {
      logger.info(`Chunk ${index + 1} details`, {
        operation: 'chunk_details',
        index: index + 1,
        content: chunk.content.substring(0, 150),
        length: chunk.content.length,
        metadata: chunk.metadata
      });
    });

    // Show cache statistics
    const cacheStats = chunker.getCacheStats();
    logger.info('Cache statistics', {
      operation: 'cache_stats',
      cacheSize: cacheStats.size,
      cacheKeys: cacheStats.keys
    });

    // Example with different model
    logger.info('--- Testing with BGE-M3 model ---', {
      operation: 'test_bge_model',
      model: 'Xenova/bge-m3'
    });
    const bgeOptions: SemanticChunkingOptions = {
      ...options,
      embeddingModel: 'Xenova/bge-m3'
    };
    
    const bgeChunker = new SemanticChunker(bgeOptions);
    const bgeChunks = await bgeChunker.chunkDocument(sampleDocument);
    
    logger.info(`BGE-M3 generated ${bgeChunks.length} chunks`, {
      operation: 'bge_results',
      chunkCount: bgeChunks.length
    });

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
      logger.info('Troubleshooting tips for semantic chunking', {
        operation: 'troubleshooting',
        tips: [
          'Ensure @huggingface/transformers is installed: npm install @huggingface/transformers',
          'Check if you have enough memory for the model',
          'Try a smaller model like "Xenova/all-MiniLM-L6-v2" first',
          'Make sure you have a stable internet connection for the first download'
        ]
      });
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
    logger.info('--- Testing convenience function ---', {
      operation: 'test_convenience_function'
    });
    const chunks = await semanticChunk(document, {
      maxChunkSize: 100,
      similarityThreshold: 0.4
    });
    
    logger.info(`Convenience function generated ${chunks.length} chunks`, {
      operation: 'convenience_function_results',
      chunkCount: chunks.length
    });
    chunks.forEach((chunk: Document, i: number) => {
      logger.info(`Convenience function chunk ${i + 1}`, {
        operation: 'convenience_chunk',
        index: i + 1,
        content: chunk.content
      });
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
  logger.info('--- TXT Loader with Semantic Chunking ---', {
    operation: 'txt_loader_demo'
  });
  
  try {
    // Load the example document using txt loader
    const documentPath = './examples/example-document.txt';
    logger.info(`Loading document from: ${documentPath}`, {
      operation: 'load_document',
      documentPath
    });
    
    const document = await loadTxt(documentPath, {
      metadata: {
        category: 'technology',
        author: 'AI Expert'
      }
    });
    
    logger.info('Document loaded successfully!', {
      operation: 'document_loaded',
      contentLength: document.metadata.contentLength,
      wordCount: document.metadata.wordCount,
      lineCount: document.metadata.lineCount,
      fileSize: document.metadata.fileSize
    });
    
    // Display some metadata
    logger.info('Document metadata', {
      operation: 'document_metadata',
      metadata: document.metadata
    });
    
    // Now perform semantic chunking on the loaded document
    logger.info('Performing semantic chunking on loaded document...', {
      operation: 'chunking_loaded_document',
      documentLength: document.content.length
    });
    const chunker = new SemanticChunker({
      maxChunkSize: 300,
      minChunkSize: 50,
      similarityThreshold: 0.3,
      embeddingModel: 'Xenova/all-MiniLM-L6-v2'
    });
    
    const chunks = await chunker.chunkDocument(document);
    logger.info(`Generated ${chunks.length} chunks from loaded document`, {
      operation: 'chunking_loaded_results',
      chunkCount: chunks.length
    });
    
    // Show first few chunks
    chunks.slice(0, 3).forEach((chunk, index) => {
      logger.info(`Preview chunk ${index + 1}`, {
        operation: 'preview_chunk',
        index: index + 1,
        content: chunk.content.substring(0, 150),
        length: chunk.content.length,
        source: chunk.metadata.source,
        chunkIndex: chunk.metadata.chunkIndex
      });
    });
    
    chunker.clearCache(true);
    
  } catch (error) {
    logger.error('Error with txt loader demonstration', error, {
      operation: 'txt_loader_error'
    });
    logger.info('Make sure example-document.txt exists in the examples directory', {
      operation: 'txt_loader_error',
      tip: 'check_file_exists'
    });
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
