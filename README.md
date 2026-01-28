# MemoryGate Demo - Trust Filtering Prevents Resurfacing

## Problem Statement

Standard RAG systems resurface corrected facts because they rely solely on semantic similarity. After a user corrects information, the old fact remains in the vector database and can still appear in top retrieval results. MemoryGate solves this by applying **trust filtering before the LLM sees context**—low-trust memories (those that have been corrected) are excluded from retrieval results entirely. This demo proves the behavioral difference between baseline RAG and MemoryGate's trust-filtered retrieval.

**Note on Baseline RAG**: The baseline demonstrates default vector-only retrieval. In practice, preventing invalid facts from resurfacing requires developers to build and maintain an external state layer (metadata filters, SQL tables, orchestration logic) alongside the vector store. This demo intentionally omits that layer to show the failure mode MemoryGate manages centrally.

## Quick Start

### 1. Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your `MEMORYGATE_API_KEY` from [https://www.memorygate.io/#request-access](https://www.memorygate.io/#request-access)

### 2. Run Demo

```bash
python examples/baseline_rag.py
python examples/with_memorygate.py
```

## Where Trust is Applied

Trust filtering occurs in the `/v1/query` API response. The MemoryGate API applies trust weighting server-side and excludes memories with `low_confidence: true` or `is_suppressed: true` from the results. **This happens before the LLM sees context**—it is not prompt weighting or post-processing. The trust filter is a pre-LLM retrieval gate that prevents corrected facts from entering the context window.

## Output Interpretation

### Baseline RAG (`baseline_rag.py`)

- Shows old fact (e.g., "123 Tech Street") in top results after correction
- Flagging a memory doesn't remove it from retrieval
- Semantic similarity alone determines ranking

### MemoryGate (`with_memorygate.py`)

- Shows old fact suppressed after correction
- Only new fact (e.g., "456 Innovation Drive") appears in results
- Trust filtering excludes low-confidence memories before LLM context

## Benchmark Scenario

The demo uses a deterministic scenario from `benchmarks/conflicting_facts.json`:

1. **Initial state**: Two policy documents (2025 and 2026) with conflicting office addresses
2. **Query**: "What is the office address?"
3. **Correction**: Flag the 2025 policy as superseded
4. **Re-query**: Compare results—baseline RAG still returns old address, MemoryGate suppresses it

## Architecture

```
Baseline RAG:
  Query → Vector Search → Top Results (includes corrected facts) → LLM Context

MemoryGate:
  Query → Vector Search → Trust Filter → Top Results (excludes corrected facts) → LLM Context
```

The trust filter is applied server-side in the `/v1/query` endpoint response. Memories with low trust scores (from corrections) are marked as `low_confidence: true` and excluded from the `results` array.

## Requirements

- Python 3.10+
- `MEMORYGATE_API_KEY` from [memorygate.io](https://www.memorygate.io/#request-access)
- Dependencies listed in `requirements.txt`

## Files

- `examples/baseline_rag.py` - Standard RAG behavior (no trust filtering)
- `examples/with_memorygate.py` - MemoryGate API integration (with trust filtering)
- `data/corpus.json` - Benchmark documents (2025/2026 policy conflict)
- `benchmarks/conflicting_facts.json` - Deterministic scenario definition
