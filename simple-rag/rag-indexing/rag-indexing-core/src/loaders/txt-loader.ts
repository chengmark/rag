import { readFileSync, existsSync } from 'fs';
import { resolve, extname } from 'path';
import { Document } from '../types/Document';
import { logger } from '../utils/logger';

/**
 * Configuration options for text loading
 */
export interface TxtLoaderOptions {
  encoding?: BufferEncoding;
  chunkSize?: number;
  preserveLineBreaks?: boolean;
  trimWhitespace?: boolean;
  removeEmptyLines?: boolean;
  metadata?: Record<string, string>;
}

/**
 * Default options for text loading
 */
const DEFAULT_OPTIONS: Required<TxtLoaderOptions> = {
  encoding: 'utf-8',
  chunkSize: 0, // 0 means don't chunk by size
  preserveLineBreaks: true,
  trimWhitespace: true,
  removeEmptyLines: false,
  metadata: {}
};

/**
 * Text file loader with support for various text formats and preprocessing
 */
export class TxtLoader {
  private options: Required<TxtLoaderOptions>;

  constructor(options: TxtLoaderOptions = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  /**
   * Load a single text file and return as a Document
   */
  async load(filePath: string): Promise<Document> {
    const resolvedPath = resolve(filePath);
    
    if (!existsSync(resolvedPath)) {
      throw new Error(`File not found: ${resolvedPath}`);
    }

    // Validate file extension
    const ext = extname(resolvedPath).toLowerCase();
    const supportedExtensions = ['.txt', '.md', '.csv', '.json', '.log', '.xml'];
    
    if (!supportedExtensions.includes(ext)) {
      throw new Error(`Unsupported file extension: ${ext}. Supported extensions: ${supportedExtensions.join(', ')}`);
    }

    try {
      // Read file content
      let content = readFileSync(resolvedPath, this.options.encoding);
      
      // Preprocess content
      content = this.preprocessContent(content);
      
      // Create metadata
      const metadata = this.createMetadata(resolvedPath, content);
      
      // Return document
      return {
        content,
        metadata
      };
      
    } catch (error) {
      throw new Error(`Failed to load file ${resolvedPath}: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Load multiple text files
   */
  async loadMultiple(filePaths: string[]): Promise<Document[]> {
    const documents: Document[] = [];
    
    for (const filePath of filePaths) {
      try {
        const document = await this.load(filePath);
        documents.push(document);
      } catch (error) {
        logger.warn(`Failed to load file ${filePath}`, {
          operation: 'load_multiple_files',
          filePath,
          error: error instanceof Error ? error.message : 'Unknown error'
        });
      }
    }
    
    return documents;
  }

  /**
   * Load text content from a string (useful for testing or direct content)
   */
  loadFromString(content: string, filename: string = 'string-input.txt'): Document {
    const processedContent = this.preprocessContent(content);
    const metadata = {
      ...this.options.metadata,
      source: filename,
      fileType: 'string-input',
      loadedAt: new Date().toISOString()
    };
    
    return {
      content: processedContent,
      metadata
    };
  }

  /**
   * Preprocess content based on options
   */
  private preprocessContent(content: string): string {
    if (this.options.trimWhitespace) {
      content = content.trim();
    }

    if (this.options.removeEmptyLines) {
      content = content.replace(/^\s*\n/gm, '');
    }

    if (!this.options.preserveLineBreaks) {
      content = content.replace(/\n+/g, ' ').replace(/\s+/g, ' ').trim();
    }

    return content;
  }

  /**
   * Create metadata for the loaded document
   */
  private createMetadata(filePath: string, content: string): Record<string, string> {
    const stats = require('fs').statSync(filePath);
    const ext = extname(filePath).toLowerCase();
    
    return {
      ...this.options.metadata,
      source: filePath,
      filename: filePath.split(/[/\\]/).pop() || '',
      fileType: ext.substring(1), // Remove the dot
      fileSize: stats.size.toString(),
      lastModified: stats.mtime.toISOString(),
      loadedAt: new Date().toISOString(),
      contentLength: content.length.toString(),
      lineCount: content.split('\n').length.toString(),
      wordCount: content.split(/\s+/).filter(word => word.length > 0).length.toString()
    };
  }

  /**
   * Get supported file extensions
   */
  static getSupportedExtensions(): string[] {
    return ['.txt', '.md', '.csv', '.json', '.log', '.xml'];
  }

  /**
   * Check if a file extension is supported
   */
  static isSupported(filePath: string): boolean {
    const ext = extname(filePath).toLowerCase();
    return this.getSupportedExtensions().includes(ext);
  }

  /**
   * Validate file before loading
   */
  static validateFile(filePath: string): { valid: boolean; errors: string[] } {
    const errors: string[] = [];
    
    if (!existsSync(filePath)) {
      errors.push('File does not exist');
    }
    
    if (!this.isSupported(filePath)) {
      errors.push(`Unsupported file extension. Supported: ${this.getSupportedExtensions().join(', ')}`);
    }
    
    try {
      const stats = require('fs').statSync(filePath);
      if (stats.size === 0) {
        errors.push('File is empty');
      }
      
      // Check if file is too large (warning for files > 10MB)
      if (stats.size > 10 * 1024 * 1024) {
        errors.push('File is very large (>10MB), consider chunking');
      }
    } catch (error) {
      errors.push('Cannot read file stats');
    }
    
    return {
      valid: errors.length === 0,
      errors
    };
  }
}

/**
 * Convenience function for loading a single text file
 */
export async function loadTxt(filePath: string, options?: TxtLoaderOptions): Promise<Document> {
  const loader = new TxtLoader(options);
  return loader.load(filePath);
}

/**
 * Convenience function for loading multiple text files
 */
export async function loadMultipleTxt(filePaths: string[], options?: TxtLoaderOptions): Promise<Document[]> {
  const loader = new TxtLoader(options);
  return loader.loadMultiple(filePaths);
}

/**
 * Convenience function for loading from string
 */
export function loadTxtFromString(content: string, filename?: string, options?: TxtLoaderOptions): Document {
  const loader = new TxtLoader(options);
  return loader.loadFromString(content, filename);
}