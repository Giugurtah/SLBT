#!/usr/bin/env python3
"""
Quick test script for the categorizer library.
Run from csrc/categorizer/ directory.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from slbt._preprocessing._backend import (
    categorize_fixed_k,
    categorize_elbow,
    categorize_silhouette,
)

def test_fixed_k():
    print("="*60)
    print("TEST: Fixed K")
    print("="*60)
    
    X = np.array([1.0, 2.0, 2.5, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0])
    print(f"Data: {X}")
    
    labels, centers = categorize_fixed_k(X, k=3)
    
    print(f"Labels:  {labels}")
    print(f"Centers: {centers}")
    print("✓ Test passed\n")

def test_elbow():
    print("="*60)
    print("TEST: Elbow Method")
    print("="*60)
    
    X = np.array([1.0, 2.0, 2.5, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0])
    print(f"Data: {X}")
    
    labels, centers, k = categorize_elbow(X, k_max=5, k_min=2)
    
    print(f"Optimal K: {k}")
    print(f"Labels:    {labels}")
    print(f"Centers:   {centers}")
    print("✓ Test passed\n")

def test_silhouette():
    print("="*60)
    print("TEST: Silhouette Method")
    print("="*60)
    
    X = np.array([1.0, 2.0, 2.5, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0])
    print(f"Data: {X}")
    
    labels, centers, k = categorize_silhouette(X, k_max=5, k_min=2)
    
    print(f"Optimal K: {k}")
    print(f"Labels:    {labels}")
    print(f"Centers:   {centers}")
    print("✓ Test passed\n")

if __name__ == "__main__":
    try:
        test_fixed_k()
        test_elbow()
        test_silhouette()
        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)