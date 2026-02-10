# PageIndex: Technical Architecture Deep Dive

## Core Problem: Why Vector RAG Fails

### 1. Query-Knowledge Mismatch
Vector similarity measures surface-level semantic relatedness, not task relevance. A query "what are debt trends?" may match documents discussing "trends" or "debt" generally, but miss sections with actual trend analysis.

**Root cause**: Embeddings encode semantic similarity, not logical relevance or domain-specific importance.

### 2. Hard Chunking Fragmentation
Fixed-size chunks (512-1000 tokens) cut through:
- Mid-sentence boundaries → incomplete thoughts
- Paragraph breaks → lost coherence
- Section transitions → missing context

**Example**: A financial statement's assets section might be split across 3 chunks, losing the relationship between current assets, long-term assets, and total calculations.

### 3. Context Window Deterioration
Research shows LLM performance degrades with context length. Retrieving 10-20 chunks creates a "needle in haystack" problem where relevant information gets buried.

### 4. Independent Query Processing
Each query treats conversation as isolated. Cannot leverage:
- Prior clarifications
- User's established focus area
- Previously retrieved context

### 5. Cross-Reference Blindness
Documents contain internal pointers: "see Appendix G", "as discussed in Section 3.2". Vector systems cannot follow these without manual preprocessing.

## PageIndex Solution: Reasoning-Based Retrieval

### Hierarchical Table of Contents Index

Replace vector databases with **in-context tree structures**:

```json
{
  "node_id": "section_2_1",
  "name": "Financial Assets",
  "description": "Overview of current and long-term financial assets including marketable securities",
  "start_index": 12,
  "end_index": 15,
  "metadata": {
    "document_type": "10-K",
    "section_type": "financial_statement"
  },
  "nodes": [
    {
      "node_id": "section_2_1_1",
      "name": "Current Assets",
      "description": "Liquid assets expected to convert to cash within one year",
      "start_index": 12,
      "end_index": 13,
      "nodes": []
    }
  ]
}
```

**Key characteristics**:
- Semantic boundaries align with document structure
- Descriptions enable reasoning ("does this section likely contain X?")
- Hierarchical nesting mirrors human navigation
- Page ranges enable precise content extraction

### Iterative Reasoning Loop

```
1. READ TABLE OF CONTENTS
   ↓ LLM reasons: "Which sections might contain relevant info?"

2. SELECT PROMISING SECTION
   ↓ Navigate tree based on descriptions

3. EXTRACT CONTENT
   ↓ Retrieve pages [start_index:end_index]

4. EVALUATE SUFFICIENCY
   ↓ LLM assesses: "Did I find what I need?"

5. BRANCH:
   - Complete → Generate answer
   - Incomplete → Loop to step 1 with refined focus
   - Cross-reference detected → Navigate to referenced section
```

### Why This Works

**Reasoning over similarity**: LLM applies domain knowledge ("financial trends require time-series data") rather than matching word patterns.

**Semantic coherence**: Retrieves complete logical units (full sections) instead of arbitrary fragments.

**Dynamic navigation**: Follows document structure like a human researcher would.

**Context awareness**: Prior iterations inform subsequent section selection.

## Implementation Architecture

### Stage 1: Indexing Pipeline

**Input**: PDF or structured markdown
**Output**: JSON tree structure

```
PDF → Extract text with structure preservation
    → Detect table of contents (first N pages)
    → Parse hierarchical headings
    → Generate LLM summaries per section
    → Construct tree with node IDs, ranges, descriptions
    → Output: index.json
```

**Configuration parameters**:
- `max-pages-per-node`: Granularity control (default: 10)
- `max-tokens-per-node`: Content limits (default: 20,000)
- `toc-check-pages`: How many initial pages to scan for ToC (default: 20)

**LLM role in indexing**:
- Infer implicit structure when explicit ToC missing
- Generate concise, semantic descriptions
- Identify section types (introduction, methodology, results, etc.)

### Stage 2: Retrieval Pipeline

**Input**: User query + conversation history + tree index
**Output**: Relevant content sections

```python
def retrieve(query: str, tree: TreeNode, history: List[str]) -> str:
    """
    Iterative reasoning-based retrieval
    """
    context = ""
    iterations = 0
    max_iterations = 5

    while iterations < max_iterations:
        # LLM reasons over tree structure
        selected_node = llm_select_node(
            query=query,
            tree=tree,
            history=history,
            prior_context=context
        )

        # Extract content from selected section
        section_content = extract_pages(
            selected_node.start_index,
            selected_node.end_index
        )

        context += section_content

        # LLM evaluates sufficiency
        decision = llm_evaluate(
            query=query,
            collected_context=context
        )

        if decision == "sufficient":
            return context
        elif decision == "follow_reference":
            # Navigate to cross-referenced section
            ref_node_id = extract_reference(section_content)
            tree = navigate_to_node(tree, ref_node_id)
        else:
            # Continue searching with refined focus
            iterations += 1

    return context
```

**Key mechanisms**:

1. **Node selection**: LLM examines node descriptions and reasons about relevance
2. **Content extraction**: Retrieve exact page ranges (preserves full context)
3. **Sufficiency evaluation**: Meta-reasoning about whether answer is possible
4. **Reference following**: Detect phrases like "Appendix G" and navigate tree

### Tree Structure Design Patterns

**Depth strategy**:
- Financial documents: 3-4 levels (Document → Section → Subsection → Item)
- Research papers: 2-3 levels (Paper → Section → Subsection)
- Technical manuals: 4-5 levels (Manual → Chapter → Section → Procedure → Step)

**Node granularity**:
- Too coarse (50+ pages): Defeats purpose, reverts to long context problem
- Too fine (1-2 pages): Excessive navigation, slower retrieval
- Optimal (5-15 pages): Semantic units, balanced coherence/efficiency

**Description quality**:
```
❌ Bad: "Section 2.1" (no semantic meaning)
✓ Good: "Balance sheet assets breakdown including current and long-term holdings"

❌ Bad: "Data analysis" (too generic)
✓ Good: "Regression analysis of customer churn factors with p-values and confidence intervals"
```

## Performance Characteristics

**Accuracy**: 98.7% on FinanceBench (financial document QA benchmark)

**Latency**: Higher per-query due to iterative reasoning (3-5 LLM calls vs 1-2 for vector RAG), but:
- No embedding computation needed
- No vector database maintenance
- No re-indexing when documents update

**Scalability**:
- Tree fits in LLM context (typically 10-50KB JSON)
- Content retrieval is random access (O(1) page extraction)
- Works for documents up to ~1000 pages

**Cost trade-offs**:
- Higher LLM API cost per query (multiple reasoning calls)
- Zero vector database hosting costs
- More accurate → fewer retries/corrections

## When to Use PageIndex Architecture

**Ideal for**:
- Long structured documents (financial filings, legal contracts, technical manuals)
- Domain-specific knowledge requiring reasoning (not just similarity)
- Documents with cross-references and internal pointers
- Multi-turn conversations building on prior context

**Not suitable for**:
- Unstructured text without clear sections (social media, chat logs)
- Similarity-based retrieval actually desired (finding "documents like this")
- Real-time streaming data (requires pre-indexed structure)
- Very short documents (<10 pages) where chunking isn't problematic
