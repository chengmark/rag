# Simple RAG

A simple RAG (Retrieval-Augmented Generation) implementation with semantic chunking, embeddings, and retrieval capabilities built as a pnpm monorepo.

## Project Structure

This is a pnpm workspace monorepo containing:

- **@rag-indexing/core** - Core RAG indexing functionality with semantic chunking, embeddings, and vector storage
- **@rag-retrieval/core** - Core RAG retrieval functionality with semantic search, similarity matching, and query processing

## Prerequisites

- Node.js >= 16.0.0
- pnpm >= 10.0.0

## Installation

### Option 1: Using the PowerShell Wrapper (Recommended)

```bash
# The project includes a pnpm.ps1 wrapper script
# Use this for all pnpm commands:

# Install all dependencies
.\pnpm.ps1 install

# Build all packages
.\pnpm.ps1 build

# Run tests
.\pnpm.ps1 test

# Development mode
.\pnpm.ps1 dev:indexing
.\pnpm.ps1 dev:retrieval
```

### Option 2: Manual pnpm Setup

```bash
# Install pnpm using the official installer
iwr https://get.pnpm.io/install.ps1 -useb | iex

# Or install via npm
npm install -g pnpm

# Install all dependencies for the monorepo
pnpm install
```

### Option 3: Using Full Path (if PATH issues)

```bash
# Use the full path to pnpm executable
C:\Users\cheng\AppData\Local\pnpm\.tools\pnpm-exe\10.33.0\pnpm.exe install
```

## Development

```bash
# Build all packages
pnpm build

# Run all tests
pnpm test

# Run tests with coverage
pnpm test:coverage

# Run linting
pnpm lint

# Fix linting issues
pnpm lint:fix

# Clean all build outputs
pnpm clean

# Development mode for indexing core
pnpm dev:indexing

# Development mode for retrieval core
pnpm dev:retrieval
```

## Package Scripts

### Root Level Scripts

- `pnpm build` - Build all packages
- `pnpm test` - Run all tests
- `pnpm test:coverage` - Run tests with coverage
- `pnpm lint` - Lint all packages
- `pnpm lint:fix` - Fix linting issues
- `pnpm clean` - Clean all build outputs
- `pnpm dev:indexing` - Development mode for @rag-indexing/core
- `pnpm dev:retrieval` - Development mode for @rag-retrieval/core

### Individual Package Scripts

You can also run scripts in individual packages:

```bash
# Run scripts in specific package
pnpm --filter @rag-indexing/core <script>
pnpm --filter @rag-retrieval/core <script>

# Examples
pnpm --filter @rag-indexing/core test
pnpm --filter @rag-retrieval-core build
```

## Workspace Configuration

The monorepo uses pnpm workspaces with the following configuration:

- **pnpm-workspace.yaml** - Defines workspace packages
- **package.json** - Root package with workspace scripts
- **workspace dependencies** - Internal packages use `workspace:*` protocol

## Usage Examples

### Indexing Example

```typescript
import { loadTxt, SemanticChunker, saveChunkEmbeddings } from '@rag-indexing/core';

// Load document
const document = await loadTxt('./document.txt');

// Semantic chunking
const chunker = new SemanticChunker();
const chunks = await chunker.chunkDocument(document);

// Generate and save embeddings
const embeddings = await generateEmbeddings(chunks);
await saveChunkEmbeddings(chunks, embeddings, './embeddings.json');
```

### Retrieval Example

```typescript
import { SearchChunkByString } from '@rag-retrieval/core';

// Initialize searcher
const searcher = new SearchChunkByString('./embeddings.json');

// Search for similar chunks
const results = await searcher.search('What is artificial intelligence?', {
  topK: 5,
  similarityThreshold: 0.3
});

console.log(results);
```

## Development Workflow

1. **Setup**: Install dependencies with `pnpm install`
2. **Development**: Use `pnpm dev:indexing` or `pnpm dev:retrieval` for development
3. **Testing**: Run `pnpm test` to run all tests
4. **Building**: Use `pnpm build` to build all packages
5. **Linting**: Run `pnpm lint` to check code quality

## Publishing

This is a private monorepo setup. For publishing, you would typically:

1. Build all packages: `pnpm build`
2. Publish individual packages:
   ```bash
   pnpm --filter @rag-indexing/core publish
   pnpm --filter @rag-retrieval/core publish
   ```

## Dependencies

### Shared Dev Dependencies

Shared development dependencies are managed at the root level and include:
- TypeScript
- Jest
- ESLint
- ts-node
- ts-node-dev

### Package Dependencies

- **@rag-indexing/core**: HuggingFace transformers, Pinecone, node-fetch
- **@rag-retrieval/core**: @rag-indexing/core (workspace dependency), node-fetch

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request
