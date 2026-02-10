# PageIndex Implementation Guide

## Building the Indexing Pipeline

### Step 1: Document Structure Detection

Extract hierarchical structure from source documents.

**For PDFs with explicit ToC**:

```python
def extract_toc_from_pdf(pdf_path: str, toc_pages: int = 20) -> List[dict]:
    """
    Parse table of contents from first N pages
    Returns list of {title, page_number, level} entries
    """
    import pdfplumber

    toc_entries = []

    with pdfplumber.open(pdf_path) as pdf:
        # Scan first N pages for ToC
        for page_num in range(min(toc_pages, len(pdf.pages))):
            page = pdf.pages[page_num]
            text = page.extract_text()

            # Detect ToC patterns:
            # - Lines with page numbers at end
            # - Indentation indicating hierarchy
            # - Common section keywords (Chapter, Section, Appendix)

            for line in text.split('\n'):
                # Pattern: "Section 2.1 Financial Assets ..... 42"
                match = re.search(r'^(\s*)(.*?)\s+\.+\s+(\d+)$', line)
                if match:
                    indent = len(match.group(1))
                    title = match.group(2).strip()
                    page_num = int(match.group(3))
                    level = estimate_level_from_indent(indent)

                    toc_entries.append({
                        'title': title,
                        'page': page_num,
                        'level': level
                    })

    return toc_entries

def estimate_level_from_indent(indent_spaces: int) -> int:
    """Convert indentation to hierarchy level (1-4)"""
    if indent_spaces == 0:
        return 1
    elif indent_spaces <= 4:
        return 2
    elif indent_spaces <= 8:
        return 3
    else:
        return 4
```

**For Markdown with heading structure**:

```python
def extract_structure_from_markdown(md_path: str) -> List[dict]:
    """
    Parse heading hierarchy from markdown
    Returns list of {title, level, line_number} entries
    """
    entries = []

    with open(md_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Match markdown headings: #, ##, ###, etc.
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                level = len(match.group(1))  # Count # symbols
                title = match.group(2).strip()

                entries.append({
                    'title': title,
                    'level': level,
                    'line_number': line_num
                })

    return entries
```

**For documents without explicit structure** (use LLM inference):

```python
def infer_structure_with_llm(document_text: str, max_tokens: int = 20000) -> List[dict]:
    """
    Use LLM to identify logical sections when ToC missing
    """
    prompt = f"""
Analyze this document and identify its logical structure.
Return a hierarchical breakdown with:
- Section titles (infer from content)
- Approximate page/position ranges
- Hierarchy level (1=main section, 2=subsection, etc.)

Document:
{document_text[:max_tokens]}

Output JSON format:
[
  {{"title": "...", "start_page": N, "end_page": M, "level": 1}},
  ...
]
"""

    response = llm_call(prompt)
    return json.loads(response)
```

### Step 2: Build Hierarchical Tree

Convert flat structure into nested tree.

```python
from typing import List, Optional
from dataclasses import dataclass, field

@dataclass
class TreeNode:
    node_id: str
    name: str
    description: str
    start_index: int
    end_index: int
    metadata: dict = field(default_factory=dict)
    nodes: List['TreeNode'] = field(default_factory=list)

def build_tree(toc_entries: List[dict], document_path: str) -> TreeNode:
    """
    Construct hierarchical tree from flat ToC entries
    """
    root = TreeNode(
        node_id="root",
        name="Document Root",
        description="Full document",
        start_index=0,
        end_index=get_total_pages(document_path)
    )

    stack = [(root, 0)]  # (node, level)

    for entry in toc_entries:
        current_level = entry['level']

        # Pop stack until we find the parent level
        while stack and stack[-1][1] >= current_level:
            stack.pop()

        parent_node = stack[-1][0] if stack else root

        # Create new node
        node = TreeNode(
            node_id=generate_node_id(entry),
            name=entry['title'],
            description="",  # Will populate with LLM later
            start_index=entry['page'],
            end_index=entry.get('end_page', entry['page'] + 10),  # Estimate
            metadata=entry.get('metadata', {})
        )

        parent_node.nodes.append(node)
        stack.append((node, current_level))

    # Fix end_index values (each section ends where next begins)
    fix_page_ranges(root)

    return root

def fix_page_ranges(node: TreeNode):
    """Adjust end indices so sections don't overlap"""
    for i, child in enumerate(node.nodes):
        if i < len(node.nodes) - 1:
            # Section ends where next section begins
            child.end_index = node.nodes[i + 1].start_index - 1
        else:
            # Last child inherits parent's end
            child.end_index = node.end_index

        # Recurse
        fix_page_ranges(child)
```

### Step 3: Generate LLM Descriptions

Add semantic descriptions to enable reasoning.

```python
def generate_descriptions(node: TreeNode, document_path: str):
    """
    Use LLM to generate semantic descriptions for each node
    """
    # Extract content for this section
    content = extract_content_range(
        document_path,
        node.start_index,
        node.end_index
    )

    prompt = f"""
Analyze this document section and write a concise description (1-2 sentences)
explaining what information it contains. Focus on:
- Key topics covered
- Type of information (data, analysis, methodology, etc.)
- Relevant domain concepts

Section title: {node.name}
Content:
{content[:2000]}  # First 2000 chars for context

Description:"""

    node.description = llm_call(prompt).strip()

    # Recurse for children
    for child in node.nodes:
        generate_descriptions(child, document_path)
```

### Step 4: Serialize Index

Save tree structure as JSON.

```python
import json

def serialize_tree(node: TreeNode) -> dict:
    """Convert TreeNode to JSON-serializable dict"""
    return {
        'node_id': node.node_id,
        'name': node.name,
        'description': node.description,
        'start_index': node.start_index,
        'end_index': node.end_index,
        'metadata': node.metadata,
        'nodes': [serialize_tree(child) for child in node.nodes]
    }

def save_index(tree: TreeNode, output_path: str):
    """Save tree index to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(serialize_tree(tree), f, indent=2)
```

## Building the Retrieval Pipeline

### Step 1: Node Selection with LLM Reasoning

```python
def select_relevant_nodes(
    query: str,
    tree: TreeNode,
    conversation_history: List[str] = None,
    max_nodes: int = 3
) -> List[TreeNode]:
    """
    Use LLM to reason about which nodes are most relevant
    """
    # Build tree representation for LLM
    tree_repr = format_tree_for_llm(tree)

    history_context = ""
    if conversation_history:
        history_context = f"""
Previous conversation:
{chr(10).join(conversation_history[-3:])}  # Last 3 turns
"""

    prompt = f"""
You are helping retrieve information from a document. Based on the query and
document structure, identify which sections are most likely to contain relevant information.

{history_context}

Current query: {query}

Document structure:
{tree_repr}

Reasoning process:
1. What type of information does the query require?
2. Which sections' descriptions indicate they contain this information?
3. Are there multiple relevant sections that should be checked?

Output the node IDs of up to {max_nodes} most promising sections, in order of priority.
Format: ["node_id_1", "node_id_2", ...]
"""

    response = llm_call(prompt)
    selected_ids = json.loads(response)

    # Convert IDs to TreeNode objects
    return [find_node_by_id(tree, nid) for nid in selected_ids]

def format_tree_for_llm(node: TreeNode, depth: int = 0) -> str:
    """
    Create readable tree representation
    """
    indent = "  " * depth
    lines = [f"{indent}- [{node.node_id}] {node.name}"]
    lines.append(f"{indent}  Description: {node.description}")
    lines.append(f"{indent}  Pages: {node.start_index}-{node.end_index}")

    for child in node.nodes:
        lines.append(format_tree_for_llm(child, depth + 1))

    return "\n".join(lines)
```

### Step 2: Content Extraction

```python
def extract_content_range(
    document_path: str,
    start_page: int,
    end_page: int
) -> str:
    """
    Extract text content from page range
    Preserves semantic boundaries
    """
    import pdfplumber

    content_chunks = []

    with pdfplumber.open(document_path) as pdf:
        for page_num in range(start_page, min(end_page + 1, len(pdf.pages))):
            page = pdf.pages[page_num]
            text = page.extract_text()

            # Preserve page boundaries for context
            content_chunks.append(f"--- Page {page_num + 1} ---\n{text}")

    return "\n\n".join(content_chunks)
```

### Step 3: Sufficiency Evaluation

```python
def evaluate_sufficiency(
    query: str,
    collected_context: str,
    max_context_length: int = 10000
) -> dict:
    """
    LLM evaluates whether collected information is sufficient to answer query
    Returns: {status: "sufficient" | "insufficient" | "follow_reference",
             reasoning: str, reference_id: Optional[str]}
    """
    prompt = f"""
Query: {query}

Collected information:
{collected_context[-max_context_length:]}  # Most recent context

Evaluation:
1. Does the collected information contain data needed to answer the query?
2. Is the answer complete, or are there gaps requiring more information?
3. Does the text reference another section (e.g., "see Appendix G", "discussed in Section 2.1")?

Output JSON:
{{
  "status": "sufficient" | "insufficient" | "follow_reference",
  "reasoning": "explanation of your assessment",
  "reference_id": "node_id if status is follow_reference, else null"
}}
"""

    response = llm_call(prompt)
    return json.loads(response)
```

### Step 4: Reference Following

```python
def follow_cross_reference(
    context: str,
    tree: TreeNode
) -> Optional[TreeNode]:
    """
    Detect cross-references like "Appendix G" or "Section 2.1" and navigate to them
    """
    # Extract reference patterns
    patterns = [
        r'see\s+(Appendix|Section|Chapter)\s+([A-Z0-9.]+)',
        r'discussed in\s+(Appendix|Section|Chapter)\s+([A-Z0-9.]+)',
        r'refer to\s+(Appendix|Section|Chapter)\s+([A-Z0-9.]+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, context, re.IGNORECASE)
        if match:
            section_type = match.group(1)
            section_id = match.group(2)

            # Find node matching this reference
            target_node = find_node_by_reference(tree, section_type, section_id)
            if target_node:
                return target_node

    return None

def find_node_by_reference(
    node: TreeNode,
    section_type: str,
    section_id: str
) -> Optional[TreeNode]:
    """
    Search tree for node matching reference
    """
    # Check current node
    if section_id.lower() in node.name.lower():
        return node

    # Recurse
    for child in node.nodes:
        result = find_node_by_reference(child, section_type, section_id)
        if result:
            return result

    return None
```

### Step 5: Complete Retrieval Loop

```python
def retrieve(
    query: str,
    tree: TreeNode,
    document_path: str,
    conversation_history: List[str] = None,
    max_iterations: int = 5
) -> str:
    """
    Complete iterative retrieval process
    """
    collected_context = ""
    visited_nodes = set()

    for iteration in range(max_iterations):
        # Select promising nodes
        candidate_nodes = select_relevant_nodes(
            query=query,
            tree=tree,
            conversation_history=conversation_history
        )

        # Extract content from unvisited nodes
        for node in candidate_nodes:
            if node.node_id not in visited_nodes:
                content = extract_content_range(
                    document_path,
                    node.start_index,
                    node.end_index
                )
                collected_context += f"\n\n=== {node.name} ===\n{content}"
                visited_nodes.add(node.node_id)

        # Evaluate sufficiency
        evaluation = evaluate_sufficiency(query, collected_context)

        if evaluation['status'] == 'sufficient':
            return collected_context

        elif evaluation['status'] == 'follow_reference':
            # Navigate to referenced section
            ref_node = follow_cross_reference(collected_context, tree)
            if ref_node and ref_node.node_id not in visited_nodes:
                content = extract_content_range(
                    document_path,
                    ref_node.start_index,
                    ref_node.end_index
                )
                collected_context += f"\n\n=== {ref_node.name} ===\n{content}"
                visited_nodes.add(ref_node.node_id)

        # Continue iteration with refined focus

    return collected_context
```

## Configuration and Tuning

### Granularity Parameters

```python
CONFIG = {
    # Indexing
    'max_pages_per_node': 10,      # Smaller = finer granularity, more navigation
    'max_tokens_per_node': 20000,  # Hard limit on node size
    'toc_check_pages': 20,         # How many pages to scan for ToC

    # Retrieval
    'max_iterations': 5,            # Prevent infinite loops
    'max_nodes_per_iteration': 3,   # How many sections to check at once
    'context_window': 10000,        # Max tokens to evaluate sufficiency

    # LLM
    'model': 'gpt-4o-2024-11-20',
    'temperature': 0.1,             # Low temp for consistent reasoning
}
```

### Quality Metrics

Monitor these indicators of index quality:

```python
def evaluate_index_quality(tree: TreeNode) -> dict:
    """
    Assess index structure quality
    """
    metrics = {
        'total_nodes': count_nodes(tree),
        'max_depth': calculate_max_depth(tree),
        'avg_pages_per_node': calculate_avg_pages(tree),
        'nodes_without_descriptions': count_missing_descriptions(tree),
        'orphaned_sections': find_orphaned_sections(tree)
    }

    # Quality checks
    warnings = []
    if metrics['avg_pages_per_node'] > 20:
        warnings.append("Nodes too coarse - consider lowering max_pages_per_node")
    if metrics['max_depth'] < 2:
        warnings.append("Tree too shallow - missing hierarchical structure")
    if metrics['nodes_without_descriptions'] > 0:
        warnings.append(f"{metrics['nodes_without_descriptions']} nodes lack descriptions")

    return {
        'metrics': metrics,
        'warnings': warnings
    }
```

## Integration Patterns

### With LangChain

```python
from langchain.schema import BaseRetriever, Document

class PageIndexRetriever(BaseRetriever):
    tree: TreeNode
    document_path: str

    def get_relevant_documents(self, query: str) -> List[Document]:
        context = retrieve(
            query=query,
            tree=self.tree,
            document_path=self.document_path
        )

        return [Document(page_content=context)]
```

### Standalone API

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    document_id: str
    history: List[str] = []

@app.post("/retrieve")
async def api_retrieve(request: QueryRequest):
    tree = load_index(request.document_id)
    document_path = get_document_path(request.document_id)

    context = retrieve(
        query=request.query,
        tree=tree,
        document_path=document_path,
        conversation_history=request.history
    )

    return {"context": context}
```
