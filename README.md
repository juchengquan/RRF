# RRF (Reciprocal Rank Fusion)

A Python library implementing various Reciprocal Rank Fusion strategies for RAG systems.

## Overview

RRF fuses ranked retrieval results from multiple retrievers (e.g., BM25, dense embeddings) to produce a unified ranking. This library handles the common RAG problem of **duplicate documents** appearing across multiple retrievers, implementing strategies that range from simple to sophisticated.

## Installation

```bash
pip install rrf
```

## Quick Start

```python
from rrf import FusionConfig, fuse_retriever_results

document_rankings = {
    "bm25": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    "dense": ["b", "a", "e", "c", "d", "f", "g", "j", "i", "h"],
    "sparse": ["c", "a", "b", "f", "d", "e", "h", "g", "j", "i"],
    "knowledge": ["d", "e", "a", "b", "c", "g", "f", "h", "i", "j"],
    "colaborative": ["i", "j", "h", "g", "f", "e", "d", "c", "b", "a"],
}

config = FusionConfig(strategy="standard")
fused = fuse_retriever_results(document_rankings, config)
# Returns: {"a": 0.079, "b": 0.079, "c": 0.078, ...} ordered by score descending
```

## Fusion Strategies

### 1. Standard RRF

Original RRF formula at document level.

```python
from rrf import standard_rrf

scores = standard_rrf(document_rankings, k=60)
```

**Formula:** $\text{score}(d) = \sum_{r} \frac{1}{k + \text{rank}_r(d)}$

**When to use:** When documents don't appear multiple times (no duplication across retrievers).

---

### 2. Best Rank Aggregation

Takes the best rank for each document across all retrievers.

```python
from rrf import best_rank_aggregation

scores = best_rank_aggregation(document_rankings, k=60)
# Returns: {"doc_id": score, ...}
```

**Formula:** $\text{score}(d) = \sum_{r} \frac{1}{k + \min(\text{rank}_r(d))}$

**When to use:** Simple, robust baseline. Avoids double-counting but throws away multi-retriever evidence.

---

### 3. Diminishing Returns

Multiple ranks contribute to the score, but with decaying weights.

```python
from rrf import diminishing_returns

# Exponential decay (default)
scores = diminishing_returns(document_rankings, alpha=0.5)

# Harmonic decay
scores = diminishing_returns(document_rankings, decay="harmonic")
```

**Formula:** $\text{score}(d) = \sum_{j} \frac{1}{k + r_j} \cdot w_j$

- Exponential: $w_j = \alpha^{j-1}$
- Harmonic: $w_j = \frac{1}{j}$

**When to use:** Recommended default for RAG. Balances single-chunk signal with multi-chunk coverage.

---

### 4. Max + Bonus

Separates primary relevance (best rank) from supporting evidence.

```python
from rrf import max_plus_bonus

scores = max_plus_bonus(document_rankings, lambda_param=0.3)
```

**Formula:** $\text{score}(d) = \frac{1}{k + r_{\min}} + \lambda \sum_{j>1} \frac{1}{k + r_j}$

**When to use:** When you want the best rank to drive ranking but still reward documents with multiple relevant ranks.

---

### 5. Soft Dedup (Rank Inflation)

Penalizes repeated documents by inflating their ranks before applying standard RRF.

```python
from rrf import soft_dedup_rank_inflation

scores = soft_dedup_rank_inflation(document_rankings, beta=0.1)
```

**Formula:** $r' = r \cdot (1 + \beta \cdot \text{occurrence index})$, then standard RRF

**When to use:** When you want to keep standard RRF behavior but encode "novelty" directly in ranks.

---

## Configuration

Use `FusionConfig` to configure fusion behavior:

```python
from rrf import FusionConfig, fuse_retriever_results

config = FusionConfig(
    strategy="diminishing_returns",  # RRFStrategy literal
    k=60,                            # RRF constant
    decay="exponential",             # "exponential" or "harmonic"
    lambda_param=0.3,               # Bonus weight for max_plus_bonus/diminishing_returns
    alpha=0.5,                       # Decay factor for exponential decay
    beta=0.1,                        # Inflation factor for soft_dedup
)

results = fuse_retriever_results(document_rankings, config)
# Returns: dict[str, float] ordered by score descending
```

### Available Strategies

| Strategy | Config Parameters |
|----------|-------------------|
| `best_rank` | `k` |
| `diminishing_returns` | `k`, `decay`, `alpha`, `lambda_param` |
| `max_plus_bonus` | `k`, `lambda_param` |
| `soft_dedup` | `k`, `beta` |
| `standard` | `k` |

## Input Format

All functions accept `document_rankings` as a dict mapping retriever names to ordered lists of document IDs:

```python
document_rankings: dict[str, list[str]] = {
    "retriever_name": ["doc_id_1", "doc_id_2", "doc_id_3", ...],
    ...
}
```

## Output Format

`fuse_retriever_results` returns an ordered dict:

```python
{"a": 0.079, "b": 0.079, "c": 0.078, ...}
```

Individual strategy functions also return a dict:

```python
{"doc_a": 0.0325, "doc_b": 0.0318, "doc_c": 0.0195, ...}
```

Both are ordered by score descending.

## Architecture

This library fits into a RAG pipeline as the **fusion layer**:

```
Sparse Retriever (BM25)
Dense Retriever (Embeddings)
         ↓
  Fusion Layer (RRF)  ← this library
         ↓
Reranker (Cross-encoder/LLM)
         ↓
   Generation
```

## Demo

Run the demo to see all strategies in action:

```bash
python examples/demo.py
```

The demo compares all strategies and shows parameter tuning effects:

```
STRATEGY COMPARISON
------------------------------------------------------------
best_rank                           a(0.079) b(0.079) c(0.078) d(0.078) e(0.078)
diminishing_returns                 a(0.009) b(0.009) c(0.009) d(0.009) e(0.009)
max_plus_bonus                      a(0.035) b(0.035) c(0.035) d(0.035) e(0.035)
soft_dedup                          a(0.079) b(0.079) c(0.078) d(0.078) e(0.078)
standard                            a(0.079) b(0.079) c(0.078) d(0.078) e(0.078)

PARAMETER TUNING
------------------------------------------------------------
Diminishing Returns (exp, α=0.3)    a(0.007) b(0.007) c(0.007) d(0.007) e(0.007)
Diminishing Returns (exp, α=0.7)    a(0.013) b(0.013) c(0.013) d(0.013) e(0.013)
Diminishing Returns (harmonic)      a(0.009) b(0.009) c(0.009) d(0.009) e(0.009)
Max + Bonus (λ=0.2)                 a(0.029) b(0.029) c(0.029) d(0.029) e(0.028)
Max + Bonus (λ=0.4)                 a(0.041) b(0.041) c(0.041) d(0.041) e(0.041)
Soft Dedup (β=0.1)                  a(0.079) b(0.079) c(0.078) d(0.078) e(0.078)
Soft Dedup (β=0.2)                  a(0.079) b(0.079) c(0.078) d(0.078) e(0.078)
```

## Type Safety

This library uses `typing.Literal` for strategy validation and `numpy.typing` for score arrays. Run type checking with:

```bash
ty check src/rrf/
```