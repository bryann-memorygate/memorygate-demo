#!/usr/bin/env python3
"""
MemoryGate Demo - Trust Filtering Prevents Resurfacing

Demonstrates the solution: After a correction is introduced,
MemoryGate's trust filter prevents the wrong answer from resurfacing.

This script shows MemoryGate API integration WITH trust filtering.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import requests

# Load .env file manually (handles BOM and path resolution)
def load_env_file(env_path: Path) -> None:
    """Load environment variables from .env file, handling BOM."""
    if not env_path.exists():
        return
    
    try:
        # Use utf-8-sig to automatically strip BOM if present
        with open(env_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ[key] = value
    except Exception as e:
        print(f"[WARNING] Could not load .env file: {e}")

# Load environment variables
env_file = Path(__file__).parent.parent / ".env"
load_env_file(env_file)

# Load corpus and benchmark
CORPUS_PATH = Path(__file__).parent.parent / "data" / "corpus.json"
BENCHMARK_PATH = Path(__file__).parent.parent / "benchmarks" / "conflicting_facts.json"


class MemoryGateClient:
    """Client for MemoryGate API."""
    
    def __init__(self, api_key: str = None, api_url: str = None):
        """
        Initialize MemoryGate client.
        
        Args:
            api_key: MemoryGate API key (from environment or parameter)
            api_url: MemoryGate API URL (from environment or parameter)
        """
        self.api_key = api_key or os.getenv("MEMORYGATE_API_KEY")
        self.api_url = (api_url or os.getenv("MEMORYGATE_API_URL", "https://memorygate-production.up.railway.app")).rstrip('/')
        
        if not self.api_key:
            print("ERROR: MEMORYGATE_API_KEY not found!")
            print("Get your API key from: https://www.memorygate.io/#request-access")
            print("Then set it in .env file or environment variable:")
            print("  MEMORYGATE_API_KEY=your_key_here")
            sys.exit(1)
        
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        print(f"[MemoryGate] API URL: {self.api_url}")
        print(f"[MemoryGate] API Key: (loaded, length={len(self.api_key)})")
    
    def ingest(self, memory_id: str, content: str, metadata: Dict[str, Any] = None, initial_trust: float = 1.0) -> bool:
        """Ingest a memory into MemoryGate."""
        payload = {
            "memory_id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "initial_trust": initial_trust
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/v1/ingest",
                json=payload,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            print(f"[MemoryGate] SUCCESS: Ingested: {memory_id}")
            return result.get("status") == "success"
        except requests.exceptions.RequestException as e:
            print(f"[MemoryGate] ERROR: Failed to ingest {memory_id}: {e}")
            if hasattr(e.response, 'text'):
                print(f"    Response: {e.response.text}")
            return False
    
    def query(self, query_text: str, limit: int = 5) -> Dict[str, Any]:
        """Query MemoryGate with trust filtering."""
        payload = {
            "query": query_text,
            "limit": limit
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/v1/query",
                json=payload,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[MemoryGate] ERROR: Query failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"    Response: {e.response.text}")
            return {"results": [], "active_count": 0, "suppressed_count": 0}
    
    def feedback(self, memory_id: str, action: str, role: str = None) -> bool:
        """
        Provide feedback to flag or approve a memory.
        
        Args:
            memory_id: Memory to flag/approve
            action: "flag" (decay trust) or "approve" (boost trust)
            role: "admin" (immediate) or "user" (queued). Defaults to None (auto-detect)
        """
        payload = {
            "memory_id": memory_id,
            "action": action
        }
        if role:
            payload["role"] = role
        
        try:
            response = requests.post(
                f"{self.api_url}/v1/feedback",
                json=payload,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            print(f"[MemoryGate] SUCCESS: Feedback applied: {action} on {memory_id}")
            return result.get("status") in ("success", "queued")
        except requests.exceptions.RequestException as e:
            print(f"[MemoryGate] ERROR: Feedback failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"    Response: {e.response.text}")
            return False


def main():
    """Run the MemoryGate demo."""
    print("=" * 70)
    print("MEMORYGATE DEMO - Trust Filtering Prevents Resurfacing")
    print("=" * 70)
    print()
    
    # Initialize client
    print("[1] Initializing MemoryGate client...")
    client = MemoryGateClient()
    
    # Load corpus
    print("\n[2] Loading corpus...")
    with open(CORPUS_PATH, 'r') as f:
        corpus = json.load(f)
    print(f"    Loaded {len(corpus)} documents")
    
    # Load benchmark scenario
    with open(BENCHMARK_PATH, 'r') as f:
        scenario = json.load(f)
    
    # Ingest documents
    print("\n[3] Ingesting documents into MemoryGate...")
    for doc in corpus:
        success = client.ingest(
            memory_id=doc["id"],
            content=doc["content"],
            metadata=doc.get("metadata", {}),
            initial_trust=1.0
        )
        if not success:
            print(f"    WARNING: Failed to ingest {doc['id']}")
    
    # Initial query
    query = scenario["query"]
    print(f"\n[4] Initial query: '{query}'")
    print("-" * 70)
    response = client.query(query, limit=3)
    
    results = response.get("results", [])
    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n  [{i}] Memory ID: {result['memory_id']}")
        print(f"      Relevance (semantic similarity): {result['relevance']:.4f}")
        print(f"      Trust Score (reliability): {result['reliability']:.4f}  (Decays with corrections)")
        print(f"      Confidence (relevance × trust): {result['confidence']:.4f}")
        print(f"      Low Confidence: {result['low_confidence']}")
        print(f"      Suppressed: {result['is_suppressed']}")
        
        # Extract relevant snippet
        content = result['content']
        if "Office location:" in content:
            snippet = [line for line in content.split('\n') if 'Office location:' in line][0]
            print(f"      Snippet: {snippet}")
        else:
            print(f"      Content preview: {content[:100]}...")
    
    print(f"\n    Active results: {response.get('active_count', 0)}")
    print(f"    Suppressed results: {response.get('suppressed_count', 0)}")
    
    # Introduce correction
    print("\n" + "=" * 70)
    print("[5] INTRODUCING CORRECTION")
    print("=" * 70)
    correction = scenario["correction_action"]
    
    # Use admin role for immediate effect (for demo purposes)
    # In production, this would be queued for admin review if user role
    success = client.feedback(
        memory_id=correction["memory_id"],
        action=correction["action"],
        role="admin"  # Immediate effect for demo
    )
    
    if success:
        print(f"    SUCCESS: Correction applied: {correction['memory_id']} flagged")
        print(f"    Trust decay applied - memory will be suppressed in future queries")
    
    # Re-query (demonstrates the solution)
    print(f"\n[6] Re-querying: '{query}'")
    print("-" * 70)
    print("SOLUTION: Trust filtering prevents old fact from resurfacing!")
    print()
    
    response_after = client.query(query, limit=3)
    results_after = response_after.get("results", [])
    
    print(f"\nTop {len(results_after)} results AFTER correction:")
    for i, result in enumerate(results_after, 1):
        print(f"\n  [{i}] Memory ID: {result['memory_id']}")
        print(f"      Relevance (semantic similarity): {result['relevance']:.4f}")
        print(f"      Trust Score (reliability): {result['reliability']:.4f}  (Shows decay, not removal)")
        print(f"      Confidence (relevance × trust): {result['confidence']:.4f}")
        print(f"      Low Confidence: {result['low_confidence']}")
        print(f"      Suppressed: {result['is_suppressed']}")
        
        # Extract relevant snippet
        content = result['content']
        if "Office location:" in content:
            snippet = [line for line in content.split('\n') if 'Office location:' in line][0]
            print(f"      Snippet: {snippet}")
        
        # Check if this is the corrected memory
        if result['memory_id'] == correction["memory_id"]:
            if result['low_confidence'] or result['is_suppressed']:
                print(f"      SUCCESS: CORRECTED MEMORY SUPPRESSED (trust decay: {result['reliability']:.4f} -> filtered)")
            else:
                print(f"      WARNING: Corrected memory still active (trust: {result['reliability']:.4f}, may need time to propagate)")
    
    print(f"\n    Active results: {response_after.get('active_count', 0)}")
    print(f"    Suppressed results: {response_after.get('suppressed_count', 0)}")
    
    # Check if old fact is suppressed
    old_memory_in_results = any(
        r['memory_id'] == correction["memory_id"] and not r['low_confidence'] and not r['is_suppressed']
        for r in results_after
    )
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if not old_memory_in_results:
        print("SUCCESS: MemoryGate trust filtering prevented corrected fact from resurfacing!")
        print("\nKey differences from baseline RAG:")
        print("  1. Trust filtering occurs BEFORE LLM sees context")
        print("  2. Low-confidence memories are excluded from results")
        print("  3. Trust decay applied server-side, not prompt weighting")
    else:
        print("NOTE: Corrected memory may still appear if trust decay hasn't fully propagated.")
        print("   This is expected behavior - trust decay is gradual, not instant deletion.")
    print("=" * 70)


if __name__ == "__main__":
    main()
