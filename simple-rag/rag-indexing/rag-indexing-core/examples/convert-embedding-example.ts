import { Document, loadTxt, saveChunkEmbeddings, searchEmbeddings, generateEmbedding } from '../src/index';
import { SemanticChunker, type SemanticChunkingOptions } from '../src/chunking/semantic-chunking';
import { 
  EmbeddingConverter, 
  EmbeddingFormat, 
  type Embedding, 
  type ConversionOptions
} from '../src/embedding/convert-embedding';
import { logger } from '../src/utils/logger';

/**
 * Example demonstrating embedding conversion with semantic chunking integration
 */

async function demonstrateEmbeddingConversionWithChunking() {
  logger.info('=== Embedding Conversion with Semantic Chunking Demo ===', {
    operation: 'demo_embedding_conversion'
  });

  // Load the example document using txt loader
  const documentPath = './examples/example-document.txt';
  logger.info(`Loading document from: ${documentPath}`, {
    operation: 'load_document',
    documentPath
  });
  
  const document = await loadTxt(documentPath, {
    metadata: {
      title: 'AI and ML Overview',
      category: 'technology',
      author: 'AI Expert'
    }
  });

  logger.info(`Loaded document: ${document.metadata.title}`, {
    operation: 'document_loaded',
    title: document.metadata.title,
    contentLength: document.metadata.contentLength,
    wordCount: document.metadata.wordCount,
    fileSize: document.metadata.fileSize
  });

  // Configure semantic chunking
  const chunkingOptions: SemanticChunkingOptions = {
    maxChunkSize: 400,
    minChunkSize: 100,
    similarityThreshold: 0.3,
    embeddingModel: 'Xenova/all-MiniLM-L6-v2',
    overlapSize: 30
  };

  try {
    // Step 1: Perform semantic chunking
    logger.info('Step 1: Performing semantic chunking...', {
      operation: 'step1_chunking',
      maxChunkSize: chunkingOptions.maxChunkSize,
      minChunkSize: chunkingOptions.minChunkSize
    });
    const chunker = new SemanticChunker(chunkingOptions);
    const chunks = await chunker.chunkDocument(document);
    
    logger.info(`Generated ${chunks.length} semantic chunks`, {
      operation: 'chunking_results',
      chunkCount: chunks.length
    });
    chunks.forEach((chunk, index) => {
      logger.info(`Chunk ${index + 1} preview`, {
        operation: 'chunk_preview',
        index: index + 1,
        content: chunk.content.substring(0, 100),
        length: chunk.content.length
      });
    });

    // Step 2: Generate embeddings for chunks in different formats
    logger.info('Step 2: Generating embeddings in different formats...');
    
    const converter = new EmbeddingConverter('Xenova/all-MiniLM-L6-v2');
    
    // Generate raw embeddings
    logger.info('--- Raw Embeddings ---', {
      operation: 'raw_embeddings',
      chunkCount: chunks.length
    });
    const rawEmbeddings: Embedding[] = [];
    for (let i = 0; i < chunks.length; i++) {
      const embedding = await converter.generateAndConvert(chunks[i].content, {
        targetFormat: EmbeddingFormat.RAW
      }) as Embedding;
      rawEmbeddings.push(embedding);
      logger.debug(`Raw embedding for chunk ${i + 1}`, {
        operation: 'raw_embedding',
        index: i + 1,
        dimensions: embedding.dimensions,
        format: embedding.format
      });
    }

    // Step 3: Convert embeddings to different formats
    logger.info('--- Format Conversions ---', {
      operation: 'format_conversions'
    });
    
    // Convert to normalized format
    logger.info('Converting to normalized format...', {
      operation: 'convert_normalized',
      embeddingCount: rawEmbeddings.length
    });
    const normalizedEmbeddings = await converter.convertEmbeddings(rawEmbeddings, {
      targetFormat: EmbeddingFormat.NORMALIZED
    });
    logger.info(`Converted ${normalizedEmbeddings.length} embeddings to normalized format`, {
      operation: 'convert_normalized',
      resultCount: normalizedEmbeddings.length
    });

    // Convert to binary format
    logger.info('Converting to binary format...', {
      operation: 'convert_binary',
      embeddingCount: rawEmbeddings.length
    });
    const binaryEmbeddings = await converter.convertEmbeddings(rawEmbeddings, {
      targetFormat: EmbeddingFormat.BINARY,
      binary: { threshold: 0 }
    });
    logger.info(`Converted ${binaryEmbeddings.length} embeddings to binary format`, {
      operation: 'convert_binary',
      resultCount: binaryEmbeddings.length
    });

    // Convert to quantized format (8-bit)
    logger.info('Converting to 8-bit quantized format...', {
      operation: 'convert_quantized',
      embeddingCount: rawEmbeddings.length
    });
    const quantizedEmbeddings = await converter.convertEmbeddings(rawEmbeddings, {
      targetFormat: EmbeddingFormat.QUANTIZED,
      quantize: { bits: 8 }
    });
    logger.info(`Converted ${quantizedEmbeddings.length} embeddings to 8-bit quantized format`, {
      operation: 'convert_quantized',
      resultCount: quantizedEmbeddings.length
    });

    // Step 4: Compare format sizes and similarities
    logger.info('--- Format Comparison ---', {
      operation: 'format_comparison'
    });
    
    [EmbeddingFormat.RAW, EmbeddingFormat.NORMALIZED, EmbeddingFormat.BINARY, EmbeddingFormat.QUANTIZED].forEach((format) => {
      const info = EmbeddingConverter.getFormatInfo(format);
      logger.debug(`Format info: ${info.name}`, {
        operation: 'format_info',
        name: info.name,
        description: info.description,
        sizeMultiplier: info.sizeMultiplier
      });
    });

    // Calculate storage savings
    const originalSize = rawEmbeddings[0].dimensions * 4; // 4 bytes per float
    const binarySize = binaryEmbeddings[0].dimensions * 0.125; // 1 bit per value
    const quantizedSize = quantizedEmbeddings[0].dimensions * 1; // 1 byte per value
    
    logger.info('Storage comparison per embedding', {
      operation: 'storage_comparison',
      rawSize: originalSize,
      binarySize: parseFloat(binarySize.toFixed(1)),
      binaryReduction: parseFloat(((1 - binarySize/originalSize) * 100).toFixed(1)),
      quantizedSize: quantizedSize,
      quantizedReduction: parseFloat(((1 - quantizedSize/originalSize) * 100).toFixed(1))
    });

    // Step 5: Similarity analysis between chunks
    logger.info('--- Chunk Similarity Analysis ---', {
      operation: 'similarity_analysis'
    });
    
    // Compare first few chunks using different formats
    const comparisons = [
      { format: 'Raw', embeddings: rawEmbeddings },
      { format: 'Normalized', embeddings: normalizedEmbeddings },
      { format: 'Binary', embeddings: binaryEmbeddings }
    ];

    for (const comparison of comparisons) {
      logger.info(`${comparison.format} format similarities`, {
        operation: 'format_similarities',
        format: comparison.format
      });
      for (let i = 0; i < Math.min(3, comparison.embeddings.length - 1); i++) {
        const similarity = converter.calculateSimilarity(
          comparison.embeddings[i],
          comparison.embeddings[i + 1],
          'cosine'
        );
        logger.debug(`Chunk similarity: ${i + 1} vs ${i + 2}`, {
          operation: 'chunk_similarity',
          format: comparison.format,
          chunk1: i + 1,
          chunk2: i + 2,
          similarity: parseFloat(similarity.toFixed(4))
        });
      }
    }

    // Step 6: Find most similar chunks
    logger.info('--- Most Similar Chunks ---', {
      operation: 'most_similar'
    });
    
    let maxSimilarity = -1;
    let mostSimilarPair: [number, number] = [0, 0];
    
    for (let i = 0; i < normalizedEmbeddings.length - 1; i++) {
      for (let j = i + 1; j < normalizedEmbeddings.length; j++) {
        const similarity = converter.calculateSimilarity(
          normalizedEmbeddings[i],
          normalizedEmbeddings[j],
          'cosine'
        );
        if (similarity > maxSimilarity) {
          maxSimilarity = similarity;
          mostSimilarPair = [i, j];
        }
      }
    }
    
    logger.info(`Most similar chunks: ${mostSimilarPair[0] + 1} and ${mostSimilarPair[1] + 1}`, {
      operation: 'most_similar_result',
      chunk1: mostSimilarPair[0] + 1,
      chunk2: mostSimilarPair[1] + 1,
      similarity: parseFloat(maxSimilarity.toFixed(4)),
      chunk1Content: chunks[mostSimilarPair[0]].content.substring(0, 80),
      chunk2Content: chunks[mostSimilarPair[1]].content.substring(0, 80)
    });

    // Step 7: Distance metrics comparison
    logger.info('--- Distance Metrics Comparison ---', {
      operation: 'distance_metrics'
    });
    
    const sampleEmbedding1 = normalizedEmbeddings[0];
    const sampleEmbedding2 = normalizedEmbeddings[1];
    
    const cosineSim = converter.calculateSimilarity(sampleEmbedding1, sampleEmbedding2, 'cosine');
    const euclideanDist = converter.calculateSimilarity(sampleEmbedding1, sampleEmbedding2, 'euclidean');
    const manhattanDist = converter.calculateSimilarity(sampleEmbedding1, sampleEmbedding2, 'manhattan');
    
    logger.info('Distance metrics between Chunk 1 and Chunk 2', {
      operation: 'distance_metrics_result',
      cosineSimilarity: parseFloat(cosineSim.toFixed(4)),
      euclideanDistance: parseFloat(euclideanDist.toFixed(4)),
      manhattanDistance: parseFloat(manhattanDist.toFixed(4))
    });

    // Step 8: Save embeddings to file for future queries
    logger.info('--- Saving Embeddings to File ---', {
      operation: 'save_embeddings'
    });
    
    try {
      // Save normalized embeddings (best for search)
      await saveChunkEmbeddings(chunks, normalizedEmbeddings, './examples/embedding-results.json');
      logger.info('Embeddings saved to ./examples/embedding-results.json', {
        operation: 'save_embeddings',
        filePath: './examples/embedding-results.json',
        embeddingCount: normalizedEmbeddings.length
      });
      
      // Demonstrate search functionality
      logger.info('--- Testing Search Functionality ---', {
        operation: 'test_search'
      });
      
      // Create a query embedding
      const queryText = "What is deep learning and neural networks?";
      logger.info(`Query: "${queryText}"`, {
        operation: 'search_query',
        query: queryText
      });
      
      const queryEmbedding = await generateEmbedding(queryText, 'Xenova/all-MiniLM-L6-v2', {
        targetFormat: EmbeddingFormat.NORMALIZED
      }) as Embedding;
      
      // Search for similar chunks
      const searchResults = await searchEmbeddings(queryEmbedding.vector, 3, './examples/embedding-results.json');
      
      logger.info(`Found ${searchResults.length} similar chunks`, {
        operation: 'search_results',
        resultCount: searchResults.length
      });
      searchResults.forEach((result, index) => {
        logger.info(`Search result ${index + 1}`, {
          operation: 'search_result',
          index: index + 1,
          similarity: parseFloat(result.similarity.toFixed(4)),
          content: result.content.substring(0, 150),
          chunkIndex: result.embedding.metadata.chunkIndex,
          model: result.embedding.model,
          format: result.embedding.format
        });
      });
      
    } catch (error) {
      logger.warn('Failed to save or search embeddings', {
        operation: 'save_search_error',
        errorMessage: error instanceof Error ? error.message : 'Unknown error'
      });
    }

    // Cleanup
    converter.cleanup();
    chunker.clearCache(true);

    logger.info('=== Demo completed successfully! ===', {
      operation: 'demo_completed'
    });

  } catch (error) {
    logger.error('Error during embedding conversion demo', error, {
      operation: 'demo_embedding_conversion'
    });
    
    if (error instanceof Error) {
      logger.error('Error details', error, {
        operation: 'error_details',
        errorMessage: error.message
      });
      logger.info('Troubleshooting tips for embedding conversion', {
        operation: 'troubleshooting',
        tips: [
          'Ensure example-document.txt exists in the examples directory',
          'Check if @huggingface/transformers is installed',
          'Verify you have enough memory for the models',
          'Try with a smaller document first'
        ]
      });
    }
  }
}

/**
 * Demonstrate convenience functions with semantic chunks
 */
async function demonstrateConvenienceFunctions() {
  logger.info('=== Convenience Functions Demo ===', {
    operation: 'demo_convenience_functions'
  });

  const sampleTexts = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing helps computers understand human language."
  ];

  try {
    logger.info('Using generateEmbedding convenience function...', {
      operation: 'convenience_function'
    });
    
    // Generate embeddings with different formats
    const rawEmbedding = await generateEmbedding(sampleTexts[0], 'Xenova/all-MiniLM-L6-v2', {
      targetFormat: EmbeddingFormat.RAW
    }) as Embedding;

    const normalizedEmbedding = await generateEmbedding(sampleTexts[0], 'Xenova/all-MiniLM-L6-v2', {
      targetFormat: EmbeddingFormat.NORMALIZED
    }) as Embedding;

    logger.info(`Generated raw embedding: ${rawEmbedding.dimensions}D, format: ${rawEmbedding.format}`, {
      operation: 'raw_embedding_result',
      dimensions: rawEmbedding.dimensions,
      format: rawEmbedding.format
    });
    logger.info(`Generated normalized embedding: ${normalizedEmbedding.dimensions}D, format: ${normalizedEmbedding.format}`, {
      operation: 'normalized_embedding_result',
      dimensions: normalizedEmbedding.dimensions,
      format: normalizedEmbedding.format
    });

    // Batch generation
    logger.info('Generating embeddings for multiple texts...', {
      operation: 'batch_generation',
      textCount: sampleTexts.length
    });
    const batchEmbeddings = await generateEmbedding(sampleTexts, 'Xenova/all-MiniLM-L6-v2', {
      targetFormat: EmbeddingFormat.NORMALIZED
    }) as Embedding[];

    logger.info(`Generated ${batchEmbeddings.length} embeddings in batch`, {
      operation: 'batch_generation',
      resultCount: batchEmbeddings.length
    });
    
    // Calculate similarities between all pairs
    logger.info('Similarities between texts', {
      operation: 'text_similarities'
    });
    for (let i = 0; i < batchEmbeddings.length; i++) {
      for (let j = i + 1; j < batchEmbeddings.length; j++) {
        const converter = new EmbeddingConverter();
        const similarity = converter.calculateSimilarity(batchEmbeddings[i], batchEmbeddings[j]);
        logger.debug(`Text similarity: ${i + 1} vs ${j + 1}`, {
          operation: 'text_similarity',
          text1: i + 1,
          text2: j + 1,
          similarity: parseFloat(similarity.toFixed(4))
        });
        converter.cleanup();
      }
    }

  } catch (error) {
    logger.error('Error with convenience functions', error, {
      operation: 'convenience_functions_error'
    });
  }
}

/**
 * Demonstrate advanced conversion scenarios
 */
async function demonstrateAdvancedConversions() {
  logger.info('=== Advanced Conversion Scenarios ===', {
    operation: 'demo_advanced_conversions'
  });

  try {
    const converter = new EmbeddingConverter();
    
    // Create a sample embedding
    const sampleEmbedding: Embedding = {
      vector: Array.from({length: 384}, () => Math.random() * 2 - 1), // Random values between -1 and 1
      dimensions: 384,
      format: EmbeddingFormat.RAW,
      model: 'Xenova/all-MiniLM-L6-v2',
      timestamp: Date.now()
    };

    logger.info('Testing advanced conversion options...', {
      operation: 'advanced_conversions'
    });

    // Custom quantization range
    logger.info('Custom quantization with specified range...', {
      operation: 'custom_quantization',
      bits: 8,
      range: [-1, 1]
    });
    const customQuantized = await converter.convertEmbedding(sampleEmbedding, {
      targetFormat: EmbeddingFormat.QUANTIZED,
      quantize: { bits: 8, range: [-1, 1] }
    });
    logger.debug(`Custom quantized embedding sample`, {
      operation: 'custom_quantization_result',
      sample: customQuantized.vector.slice(0, 5).join(', '),
      bits: 8
    });

    // Binary with custom threshold
    logger.info('Binary conversion with custom threshold...', {
      operation: 'custom_binary',
      threshold: 0.1
    });
    const customBinary = await converter.convertEmbedding(sampleEmbedding, {
      targetFormat: EmbeddingFormat.BINARY,
      binary: { threshold: 0.1 }
    });
    const onesCount = customBinary.vector.filter(v => v === 1).length;
    logger.info(`Binary embedding (threshold 0.1) results`, {
      operation: 'custom_binary_result',
      threshold: 0.1,
      onesCount,
      zerosCount: customBinary.vector.length - onesCount
    });

    // Chain conversions
    logger.info('Chaining multiple conversions...', {
      operation: 'chain_conversions'
    });
    const normalized = await converter.convertEmbedding(sampleEmbedding, {
      targetFormat: EmbeddingFormat.NORMALIZED
    });
    const binaryFromNormalized = await converter.convertEmbedding(normalized, {
      targetFormat: EmbeddingFormat.BINARY
    });
    logger.info('Chain conversion: RAW -> NORMALIZED -> BINARY completed', {
      operation: 'chain_conversion_completed'
    });

    converter.cleanup();

  } catch (error) {
    logger.error('Error in advanced conversions', error, {
      operation: 'advanced_conversions_error'
    });
  }
}

// Export for use in other modules
export { 
  demonstrateEmbeddingConversionWithChunking,
  demonstrateConvenienceFunctions,
  demonstrateAdvancedConversions
};

// Run if this file is executed directly
if (require.main === module) {
  (async () => {
    await demonstrateEmbeddingConversionWithChunking();
    await demonstrateConvenienceFunctions();
    await demonstrateAdvancedConversions();
  })().catch(error => logger.error('Unhandled error in embedding conversion demo', error, {
    operation: 'main'
  }));
}