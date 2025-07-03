#!/usr/bin/env python3
"""
Test script for TreeQuest MCP Server

This script demonstrates how to use the TreeQuest MCP server tools.
"""

import sys
import json
from pathlib import Path

# Add the src directory to Python path to import treequest
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from fastmcp import FastMCP
        print("✓ fastMCP imported successfully")
    except ImportError as e:
        print(f"✗ fastMCP import failed: {e}")
        return False
    
    try:
        import treequest as tq
        print("✓ TreeQuest imported successfully")
        print(f"  Available algorithms: {tq.__all__}")
    except ImportError as e:
        print(f"✗ TreeQuest import failed: {e}")
        return False
    
    return True

def test_algorithms():
    """Test that TreeQuest algorithms can be instantiated."""
    print("\nTesting TreeQuest algorithms...")
    
    try:
        import treequest as tq
        
        # Test StandardMCTS
        algo = tq.StandardMCTS()
        print("✓ StandardMCTS created successfully")
        
        # Test ABMCTSA
        algo = tq.ABMCTSA()
        print("✓ ABMCTSA created successfully")
        
        # Test AB-MCTS-M (requires extra dependencies)
        try:
            algo = tq.ABMCTSM()
            print("✓ ABMCTSM created successfully")
        except Exception as e:
            print(f"⚠ ABMCTSM creation failed (may need additional deps): {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Algorithm test failed: {e}")
        return False

def test_simple_search():
    """Test a simple search with TreeQuest."""
    print("\nTesting simple tree search...")
    
    try:
        import treequest as tq
        import random
        
        # Define a simple generation function
        def generate(parent_state):
            if parent_state is None:
                new_state = "root"
            else:
                new_state = f"child_of_{parent_state}"
            
            score = random.random()
            return new_state, score
        
        # Create algorithm and tree
        algo = tq.StandardMCTS()
        search_tree = algo.init_tree()
        
        # Run a few search steps
        for _ in range(5):
            search_tree = algo.step(search_tree, {'action': generate})
        
        # Get results
        results = tq.top_k(search_tree, algo, k=3)
        print(f"✓ Search completed. Top 3 results:")
        for i, (state, score) in enumerate(results, 1):
            print(f"  {i}. State: {state}, Score: {score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Simple search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("TreeQuest MCP Server Test Suite")
    print("=" * 40)
    
    all_passed = True
    
    # Test imports
    all_passed &= test_imports()
    
    # Test algorithms
    all_passed &= test_algorithms() 
    
    # Test simple search
    all_passed &= test_simple_search()
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All tests passed! TreeQuest MCP server should work correctly.")
    else:
        print("✗ Some tests failed. Check the error messages above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())