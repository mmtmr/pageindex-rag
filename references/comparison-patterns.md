# Comparing Vector RAG vs PageIndex Architecture

## When to Choose Each Approach

### Use Vector RAG when:

1. **Unstructured, heterogeneous content**
   - Social media posts, customer reviews, chat logs
   - Mixed content types without clear hierarchical structure
   - Short, independent documents (<5 pages)

2. **Similarity-based tasks are actually desired**
   - "Find documents similar to this one"
   - Semantic search across diverse content
   - Exploratory research without specific information needs

3. **Real-time or streaming data**
   - Continuous content ingestion
   - Live updates without re-indexing entire corpus
   - Dynamic collections where documents frequently added/removed

4. **Limited LLM API budget**
   - Vector search is O(1) cost per query
   - PageIndex requires multiple LLM reasoning calls
   - High query volume with cost constraints

### Use PageIndex when:

1. **Long structured documents**
   - Financial reports (10-Ks, annual reports)
   - Legal contracts and agreements
   - Technical manuals and specifications
   - Research papers with clear sections

2. **Domain-specific precision required**
   - Questions require reasoning, not just similarity
   - Answers depend on understanding document structure
   - Context matters (introduction vs methodology vs results)

3. **Cross-references are critical**
   - Documents with internal pointers ("see Appendix G")
   - Related sections that should be retrieved together
   - Hierarchical information (parent-child relationships)

4. **Multi-turn conversations**
   - Users refine queries based on prior answers
   - Building context over conversation
   - Need to remember which sections already explored

## Architectural Trade-offs

| Dimension | Vector RAG | PageIndex |
|-----------|-----------|-----------|
| **Setup complexity** | Medium (need embedding model + vector DB) | Low (just LLM + JSON) |
| **Indexing cost** | High (embed all chunks) | Medium (LLM summaries per section) |
| **Query cost** | Low (vector similarity) | High (multiple LLM calls) |
| **Query latency** | Fast (milliseconds) | Slower (seconds) |
| **Accuracy on structured docs** | 60-80% | 95-99% |
| **Accuracy on unstructured docs** | 70-90% | 60-80% |
| **Storage requirements** | Large (vector DB) | Minimal (JSON index) |
| **Maintenance** | Complex (vector DB ops) | Simple (JSON files) |
| **Context preservation** | Poor (fixed chunks) | Excellent (semantic sections) |
| **Cross-reference handling** | Fails | Succeeds |

## Hybrid Approaches

### Combine both for different query types

```python
def hybrid_retrieve(query: str, query_type: str) -> str:
    """
    Route to appropriate retrieval method based on query characteristics
    """
    if query_type == "factual":
        # Use PageIndex for precise information retrieval
        return pageindex_retrieve(query)

    elif query_type == "exploratory":
        # Use vector search for broad similarity
        return vector_retrieve(query)

    elif query_type == "analytical":
        # Use PageIndex to gather data, then analyze
        context = pageindex_retrieve(query)
        return analyze_with_llm(context, query)
```

### Two-stage retrieval

```python
def two_stage_retrieve(query: str) -> str:
    """
    Stage 1: Vector search narrows document set
    Stage 2: PageIndex navigates within selected documents
    """
    # Stage 1: Which documents are relevant?
    relevant_docs = vector_search(query, top_k=3)

    # Stage 2: Within each document, use PageIndex
    contexts = []
    for doc in relevant_docs:
        tree = load_pageindex_tree(doc.id)
        context = pageindex_retrieve(query, tree, doc.path)
        contexts.append(context)

    return "\n\n".join(contexts)
```

## Migration Strategies

### From Vector RAG to PageIndex

**Step 1**: Identify problematic document types
- Long documents with poor retrieval accuracy
- Documents with cross-references
- Structured documents (financial, legal, technical)

**Step 2**: Build PageIndex for high-value documents
```python
# Start with most important documents
priority_docs = [
    "10-K_annual_report.pdf",
    "technical_specification.pdf",
    "user_manual.pdf"
]

for doc_path in priority_docs:
    tree = build_pageindex(doc_path)
    save_index(tree, f"{doc_path}.index.json")
```

**Step 3**: A/B test retrieval quality
```python
def compare_retrievers(query: str, doc_path: str):
    # Vector retrieval
    vector_result = vector_retrieve(query, doc_path)

    # PageIndex retrieval
    tree = load_index(doc_path)
    pageindex_result = pageindex_retrieve(query, tree, doc_path)

    # Compare results
    return {
        'vector': vector_result,
        'pageindex': pageindex_result,
        'user_preference': ask_user_which_better()
    }
```

**Step 4**: Gradually expand coverage
- Measure accuracy improvements
- Monitor cost impact (more LLM calls vs better results)
- Roll out to more document types

### Combining with existing LangChain pipelines

```python
from langchain.schema import BaseRetriever
from langchain.chains import RetrievalQA

class HybridRetriever(BaseRetriever):
    """
    Use PageIndex for structured docs, vector search for others
    """
    def __init__(self, vector_store, pageindex_trees):
        self.vector_store = vector_store
        self.pageindex_trees = pageindex_trees

    def get_relevant_documents(self, query: str):
        # Detect document type from query context
        doc_type = classify_document_type(query)

        if doc_type in self.pageindex_trees:
            # Use PageIndex for structured documents
            tree = self.pageindex_trees[doc_type]
            context = pageindex_retrieve(query, tree)
            return [Document(page_content=context)]
        else:
            # Fall back to vector search
            return self.vector_store.similarity_search(query)

# Use in QA chain
retriever = HybridRetriever(
    vector_store=vector_db,
    pageindex_trees={
        'financial': load_index('10k.index.json'),
        'legal': load_index('contract.index.json')
    }
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)
```

## Common Pitfalls and Solutions

### Pitfall 1: Over-fragmenting with small nodes

**Problem**: Setting `max_pages_per_node=1` creates too much navigation overhead.

**Solution**: Use 5-15 pages per node. Test with sample queries to find optimal granularity.

```python
# Test different granularities
for pages_per_node in [5, 10, 15, 20]:
    tree = build_index(doc_path, max_pages=pages_per_node)
    avg_iterations = benchmark_queries(test_queries, tree)
    print(f"Pages per node: {pages_per_node}, Avg iterations: {avg_iterations}")
```

### Pitfall 2: Poor description quality

**Problem**: Vague descriptions ("Section 2.1") don't enable reasoning.

**Solution**: Use LLM to generate semantic descriptions. Include domain keywords.

```python
# Bad
node.description = node.name  # Just copies title

# Good
prompt = f"""
Describe what information this section contains, focusing on:
- Key topics and concepts
- Type of data (quantitative, qualitative, procedural)
- Relevant domain terminology

Section: {node.name}
Content preview: {content[:2000]}
"""
node.description = llm_call(prompt)
```

### Pitfall 3: Not handling missing ToC

**Problem**: Many PDFs lack explicit table of contents.

**Solution**: Use LLM to infer structure from content patterns.

```python
def detect_structure_strategy(doc_path: str) -> str:
    """
    Determine best approach for extracting structure
    """
    first_20_pages = extract_pages(doc_path, 0, 20)

    # Check for explicit ToC
    if has_table_of_contents(first_20_pages):
        return "explicit_toc"

    # Check for markdown-style headings
    if has_heading_hierarchy(doc_path):
        return "heading_based"

    # Fall back to LLM inference
    return "llm_inference"
```

### Pitfall 4: Ignoring conversation history

**Problem**: Each query treated independently, missing context refinement.

**Solution**: Pass conversation history to node selection.

```python
def select_nodes_with_history(
    query: str,
    tree: TreeNode,
    history: List[dict]  # [{query: str, selected_nodes: List[str]}]
) -> List[TreeNode]:
    """
    Use prior queries to inform section selection
    """
    # Extract context from history
    prior_sections = [
        node_id
        for turn in history
        for node_id in turn['selected_nodes']
    ]

    prompt = f"""
Previous queries explored: {prior_sections}
Current query: {query}

Should we:
1. Continue in same section (refinement)
2. Move to related section (expansion)
3. Check entirely different section (pivot)

Which nodes should we examine?
"""

    return llm_select_nodes(prompt, tree)
```

## Performance Optimization

### Caching strategies

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def extract_content_cached(doc_path: str, start: int, end: int) -> str:
    """
    Cache extracted content to avoid re-reading pages
    """
    return extract_content_range(doc_path, start, end)

@lru_cache(maxsize=10)
def load_index_cached(index_path: str) -> TreeNode:
    """
    Keep frequently accessed indices in memory
    """
    return load_index(index_path)
```

### Parallel node evaluation

```python
import asyncio

async def evaluate_nodes_parallel(
    query: str,
    candidate_nodes: List[TreeNode]
) -> List[Tuple[TreeNode, float]]:
    """
    Score multiple nodes in parallel
    """
    async def score_node(node: TreeNode) -> Tuple[TreeNode, float]:
        score = await llm_evaluate_relevance(query, node)
        return (node, score)

    results = await asyncio.gather(
        *[score_node(node) for node in candidate_nodes]
    )

    # Sort by score
    return sorted(results, key=lambda x: x[1], reverse=True)
```

### Early stopping

```python
def retrieve_with_early_stop(
    query: str,
    tree: TreeNode,
    confidence_threshold: float = 0.9
) -> str:
    """
    Stop iteration when high confidence reached
    """
    context = ""

    for iteration in range(max_iterations):
        nodes = select_nodes(query, tree)
        context += extract_content(nodes)

        # Check confidence
        confidence = evaluate_confidence(query, context)

        if confidence > confidence_threshold:
            return context  # Early stop

    return context
```
