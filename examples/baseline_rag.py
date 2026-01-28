#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline RAG Demo - Standard RAG Behavior

Demonstrates the problem: After a correction is introduced, 
the wrong answer still resurfaces in retrieval results.

This script shows standard RAG behavior WITHOUT trust filtering.
"""

import json
import os
import sys
import io
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

# Fix Windows console encoding for Unicode/emoji support
if sys.platform == 'win32':
    try:
        # Try to reconfigure stdout/stderr to UTF-8
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        # Fallback for older Python versions - wrap stdout/stderr
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Load corpus
CORPUS_PATH = Path(__file__).parent.parent / "data" / "corpus.json"
BENCHMARK_PATH = Path(__file__).parent.parent / "benchmarks" / "conflicting_facts.json"


class BaselineRAG:
    """Simple RAG system with no trust filtering."""
    
    def __init__(self):
        """Initialize with embedding model."""
        print("[Baseline RAG] Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.memories: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = None
        
    def add_memory(self, memory_id: str, content: str, metadata: Dict[str, Any] = None):
        """Add a memory to the system."""
        embedding = self.embedder.encode(content, normalize_embeddings=True)
        
        memory = {
            "memory_id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "embedding": embedding,
            "trust_weight": 1.0  # All memories start with full trust
        }
        
        self.memories.append(memory)
        
        # Update embeddings matrix
        if self.embeddings is None:
            self.embeddings = np.array([embedding])
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        
        print(f"[Baseline RAG] Added memory: {memory_id}")
    
    def query(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Query the system - naive retrieval with no trust filtering."""
        if not self.memories:
            return []
        
        # Generate query embedding
        query_embedding = self.embedder.encode(query_text, normalize_embeddings=True)
        
        # Compute cosine similarities (normalized embeddings, so dot product = cosine similarity)
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top N results
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        results = []
        for idx in top_indices:
            memory = self.memories[idx].copy()
            memory["relevance"] = float(similarities[idx])
            memory.pop("embedding", None)  # Remove embedding from output
            results.append(memory)
        
        return results
    
    def flag_memory(self, memory_id: str, reason: str = None):
        """Flag a memory as incorrect (but still returns it in queries)."""
        for memory in self.memories:
            if memory["memory_id"] == memory_id:
                memory["trust_weight"] = 0.0  # Set trust to 0, but still retrievable
                memory["flagged"] = True
                print(f"[Baseline RAG] Flagged memory: {memory_id} (reason: {reason})")
                print(f"[Baseline RAG] ⚠️  WARNING: Memory still appears in query results!")
                return
        print(f"[Baseline RAG] Memory not found: {memory_id}")


def main():
    """Run the baseline RAG demo."""
    print("=" * 70)
    print("BASELINE RAG DEMO - Standard RAG Behavior")
    print("=" * 70)
    print()
    
    # Load corpus
    print("[1] Loading corpus...")
    with open(CORPUS_PATH, 'r') as f:
        corpus = json.load(f)
    print(f"    Loaded {len(corpus)} documents")
    
    # Load benchmark scenario
    with open(BENCHMARK_PATH, 'r') as f:
        scenario = json.load(f)
    
    # Initialize RAG system
    print("\n[2] Initializing baseline RAG system...")
    rag = BaselineRAG()
    
    # Ingest documents
    print("\n[3] Ingesting documents...")
    for doc in corpus:
        rag.add_memory(
            memory_id=doc["id"],
            content=doc["content"],
            metadata=doc.get("metadata", {})
        )
    
    # Initial query
    query = scenario["query"]
    print(f"\n[4] Initial query: '{query}'")
    print("-" * 70)
    results = rag.query(query, n_results=3)
    
    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n  [{i}] Memory ID: {result['memory_id']}")
        print(f"      Relevance: {result['relevance']:.4f}")
        print(f"      Trust Weight: {result['trust_weight']:.2f}")
        # Extract relevant snippet
        content = result['content']
        if "Office location:" in content:
            snippet = [line for line in content.split('\n') if 'Office location:' in line][0]
            print(f"      Snippet: {snippet}")
        else:
            print(f"      Content preview: {content[:100]}...")
    
    # Introduce correction
    print("\n" + "=" * 70)
    print("[5] INTRODUCING CORRECTION")
    print("=" * 70)
    correction = scenario["correction_action"]
    rag.flag_memory(
        memory_id=correction["memory_id"],
        reason=correction["reason"]
    )
    
    # Re-query (demonstrates the problem)
    print(f"\n[6] Re-querying: '{query}'")
    print("-" * 70)
    print("⚠️  PROBLEM: Even after flagging, the old fact still appears!")
    print()
    
    results_after = rag.query(query, n_results=3)
    
    print(f"\nTop {len(results_after)} results AFTER correction:")
    for i, result in enumerate(results_after, 1):
        print(f"\n  [{i}] Memory ID: {result['memory_id']}")
        print(f"      Relevance: {result['relevance']:.4f}")
        print(f"      Trust Weight: {result['trust_weight']:.2f}")
        print(f"      Flagged: {result.get('flagged', False)}")
        
        # Extract relevant snippet
        content = result['content']
        if "Office location:" in content:
            snippet = [line for line in content.split('\n') if 'Office location:' in line][0]
            print(f"      Snippet: {snippet}")
        
        if result.get('flagged'):
            print(f"      WARNING: WRONG ANSWER STILL IN RESULTS!")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("Baseline RAG returns corrected facts because:")
    print("  1. No trust filtering before retrieval")
    print("  2. Flagging doesn't remove memories from results")
    print("  3. Semantic similarity alone determines ranking")
    print()
    print("This is why MemoryGate's pre-LLM trust filter is needed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
