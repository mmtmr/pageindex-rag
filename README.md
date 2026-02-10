# PageIndex RAG Skill

> Reasoning-based retrieval architecture replacing vector databases with hierarchical navigation

## Overview

PageIndex replaces vector-based similarity search with LLM-driven hierarchical navigation, achieving **98.7% accuracy** on financial document benchmarks by reasoning through document structure instead of matching embeddings.

## Installation

```bash
npx skills add mmtmr/pageindex-rag -g -y
```

## What This Skill Provides

- **RAG Architecture Design**: Hierarchical table-of-contents indices with LLM-driven navigation
- **Implementation Patterns**: Practical code examples for building PageIndex systems
- **Vector RAG Migration**: Converting existing vector-based systems to reasoning-based retrieval
- **Document Indexing**: Strategies for structured documents (financial reports, legal contracts, technical manuals)

## Use Cases

Use this skill when:
1. Implementing RAG for long structured documents (financial reports, legal contracts, technical manuals)
2. Improving existing vector-based RAG systems with poor accuracy on structured content
3. Designing document indexing strategies with semantic coherence
4. Explaining PageIndex concepts including reasoning-based retrieval, hierarchical navigation, and cross-reference following
5. Handling documents with internal references and multi-turn conversations

## Core Innovation

**Why Vector RAG Fails:**
- Query-knowledge mismatch (surface semantics ≠ task relevance)
- Hard chunking fragments contextual continuity
- Context window deterioration with 10-20 chunks
- Cannot follow cross-references

**PageIndex Solution:**
Replace vector databases with hierarchical tree indices stored as JSON, enabling LLM-driven navigation through document structure.

## Key Features

- 98.7% accuracy on financial document benchmarks
- Hierarchical navigation with semantic coherence
- Cross-reference following capability
- Multi-turn conversation support
- No embedding dependency

## References

- [Architecture Overview](references/architecture.md)
- [Comparison Patterns](references/comparison-patterns.md)
- [Implementation Guide](references/implementation-guide.md)

## License

MIT

## Author

Created by [@mmtmr](https://github.com/mmtmr)
