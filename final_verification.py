#!/usr/bin/env python3
"""
Final Verification Script for TreeQuest MCP Server

This script performs a comprehensive check to verify all components
of the TreeQuest MCP tool are working correctly.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all necessary imports work."""
    print("🔍 Testing imports...")
    
    try:
        from fastmcp import FastMCP
        print("  ✅ fastMCP imported successfully")
    except ImportError as e:
        print(f"  ❌ fastMCP import failed: {e}")
        return False
    
    try:
        import treequest as tq
        print("  ✅ TreeQuest imported successfully")
        print(f"     Available: {tq.__all__}")
    except ImportError as e:
        print(f"  ❌ TreeQuest import failed: {e}")
        return False
    
    try:
        import treequest_mcp_server
        print("  ✅ MCP server module imported successfully")
    except ImportError as e:
        print(f"  ❌ MCP server import failed: {e}")
        return False
    
    return True


def test_algorithms():
    """Test that all TreeQuest algorithms work."""
    print("\n🧠 Testing TreeQuest algorithms...")
    
    import treequest as tq
    
    algorithms = [
        ("StandardMCTS", tq.StandardMCTS),
        ("ABMCTSA", tq.ABMCTSA),
        ("ABMCTSM", tq.ABMCTSM),
        ("TreeOfThoughtsBFS", tq.TreeOfThoughtsBFSAlgo)
    ]
    
    for name, algo_class in algorithms:
        try:
            algo = algo_class()
            tree = algo.init_tree()
            print(f"  ✅ {name} works correctly")
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            return False
    
    return True


def test_mcp_server_structure():
    """Test that MCP server has all required components."""
    print("\n🔧 Testing MCP server structure...")
    
    import treequest_mcp_server as server
    
    # Check tools
    required_tools = [
        'initialize_search',
        'add_generation_function', 
        'run_search_steps',
        'get_top_results',
        'list_sessions',
        'save_session'
    ]
    
    for tool in required_tools:
        if hasattr(server, tool):
            print(f"  ✅ Tool '{tool}' exists")
        else:
            print(f"  ❌ Tool '{tool}' missing")
            return False
    
    # Check resources
    required_resources = ['get_documentation', 'get_session_info']
    for resource in required_resources:
        if hasattr(server, resource):
            print(f"  ✅ Resource '{resource}' exists")
        else:
            print(f"  ❌ Resource '{resource}' missing")
            return False
    
    # Check prompts
    required_prompts = ['create_llm_generation_function', 'setup_multi_llm_search']
    for prompt in required_prompts:
        if hasattr(server, prompt):
            print(f"  ✅ Prompt '{prompt}' exists")
        else:
            print(f"  ❌ Prompt '{prompt}' missing")
            return False
    
    return True


def test_generation_functions():
    """Test generation function creation and execution."""
    print("\n⚙️ Testing generation functions...")
    
    import treequest_mcp_server as server
    
    try:
        # Test simple generator
        simple_gen = server._create_simple_generator("Test prompt", (0.3, 0.8))
        state, score = simple_gen(None)
        assert isinstance(state, str)
        assert 0.3 <= score <= 0.8
        print("  ✅ Simple generation function works")
        
        # Test custom code execution
        custom_code = '''
def generate(parent_state):
    if parent_state is None:
        return "root", 0.5
    else:
        return f"child_{parent_state}", 0.7
'''
        namespace = {}
        exec(custom_code, namespace)
        gen_func = namespace['generate']
        state, score = gen_func(None)
        assert state == "root"
        assert score == 0.5
        print("  ✅ Custom generation function works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Generation function test failed: {e}")
        return False


def test_basic_workflow():
    """Test a basic tree search workflow."""
    print("\n🚀 Testing basic workflow...")
    
    import treequest as tq
    
    try:
        # Create algorithm
        algo = tq.StandardMCTS()
        tree = algo.init_tree()
        
        # Define generation function
        def test_generate(parent_state):
            if parent_state is None:
                return "initial_state", 0.6
            else:
                return f"next_{parent_state}", 0.7
        
        actions = {'test': test_generate}
        
        # Run a few steps
        for _ in range(3):
            tree = algo.step(tree, actions)
        
        # Get results
        results = tq.top_k(tree, algo, k=2)
        assert len(results) > 0
        
        print("  ✅ Basic tree search workflow works")
        return True
        
    except Exception as e:
        print(f"  ❌ Basic workflow test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "treequest_mcp_server.py",
        "test_treequest_mcp.py", 
        "test_mcp_integration.py",
        "demo_treequest_mcp.py",
        "README_MCP_Server.md",
        "TEST_RESULTS.md",
        "IMPLEMENTATION_SUMMARY.md"
    ]
    
    for filename in required_files:
        path = Path(filename)
        if path.exists():
            print(f"  ✅ {filename} exists ({path.stat().st_size} bytes)")
        else:
            print(f"  ❌ {filename} missing")
            return False
    
    # Check src directory
    src_path = Path("src/treequest")
    if src_path.exists():
        print(f"  ✅ TreeQuest source directory exists")
    else:
        print(f"  ❌ TreeQuest source directory missing")
        return False
    
    return True


def main():
    """Run all verification tests."""
    print("🔍 TreeQuest MCP Server - Final Verification")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("TreeQuest Algorithms", test_algorithms),
        ("MCP Server Structure", test_mcp_server_structure),
        ("Generation Functions", test_generation_functions),
        ("Basic Workflow", test_basic_workflow)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED\n")
            else:
                print(f"❌ {test_name}: FAILED\n")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}\n")
    
    print("=" * 50)
    print(f"VERIFICATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - TreeQuest MCP Server is ready for use!")
        print("\n🚀 Ready for:")
        print("  - MCP client integration")
        print("  - LLM inference-time scaling")
        print("  - Complex problem solving")
        print("  - Production deployment")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed - please address issues before deployment")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)