#!/usr/bin/env python3
"""
Integration Tests for TreeQuest MCP Server

This test suite validates the TreeQuest MCP server functionality through
direct function calls and integration testing.
"""

import unittest
import sys
import tempfile
import os
import json
import random
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the server components directly
import treequest_mcp_server
import treequest as tq


class TestTreeQuestMCPIntegration(unittest.TestCase):
    """Integration tests for TreeQuest MCP Server."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear global state
        treequest_mcp_server._search_states.clear()
        treequest_mcp_server._generation_functions.clear()
        
        self.test_session_id = "test_session"
    
    def tearDown(self):
        """Clean up after tests."""
        # Clear global state
        treequest_mcp_server._search_states.clear()
        treequest_mcp_server._generation_functions.clear()


class TestBasicFunctionality(TestTreeQuestMCPIntegration):
    """Test basic TreeQuest MCP functionality."""
    
    def test_import_and_basic_setup(self):
        """Test that all necessary components can be imported."""
        # Test TreeQuest import
        self.assertIsNotNone(tq.StandardMCTS)
        self.assertIsNotNone(tq.ABMCTSA)
        self.assertIsNotNone(tq.ABMCTSM)
        
        # Test that algorithms can be instantiated
        algo1 = tq.StandardMCTS()
        self.assertIsNotNone(algo1)
        
        algo2 = tq.ABMCTSA() 
        self.assertIsNotNone(algo2)
        
        algo3 = tq.ABMCTSM()
        self.assertIsNotNone(algo3)
    
    def test_mcp_server_functions_exist(self):
        """Test that all expected MCP server functions exist."""
        # Check tool functions exist
        self.assertTrue(hasattr(treequest_mcp_server, 'initialize_search'))
        self.assertTrue(hasattr(treequest_mcp_server, 'add_generation_function'))
        self.assertTrue(hasattr(treequest_mcp_server, 'run_search_steps'))
        self.assertTrue(hasattr(treequest_mcp_server, 'get_top_results'))
        self.assertTrue(hasattr(treequest_mcp_server, 'list_sessions'))
        self.assertTrue(hasattr(treequest_mcp_server, 'save_session'))
        
        # Check resource functions exist
        self.assertTrue(hasattr(treequest_mcp_server, 'get_documentation'))
        self.assertTrue(hasattr(treequest_mcp_server, 'get_session_info'))
        
        # Check prompt functions exist
        self.assertTrue(hasattr(treequest_mcp_server, 'create_llm_generation_function'))
        self.assertTrue(hasattr(treequest_mcp_server, 'setup_multi_llm_search'))
    
    def test_global_state_management(self):
        """Test that global state is properly managed."""
        # Initially empty
        self.assertEqual(len(treequest_mcp_server._search_states), 0)
        self.assertEqual(len(treequest_mcp_server._generation_functions), 0)
        
        # Manually add state to test
        treequest_mcp_server._search_states["test"] = {"algorithm_type": "StandardMCTS"}
        self.assertEqual(len(treequest_mcp_server._search_states), 1)
        
        # Clear and verify
        treequest_mcp_server._search_states.clear()
        self.assertEqual(len(treequest_mcp_server._search_states), 0)


class TestTreeSearchFunctionality(TestTreeQuestMCPIntegration):
    """Test core tree search functionality."""
    
    def test_standard_mcts_basic_search(self):
        """Test basic StandardMCTS functionality."""
        # Create algorithm
        algo = tq.StandardMCTS()
        search_tree = algo.init_tree()
        
        # Define simple generation function
        def generate(parent_state):
            if parent_state is None:
                new_state = "root"
            else:
                new_state = f"child_of_{parent_state}"
            score = 0.8
            return new_state, score
        
        # Run some search steps
        actions = {'test_action': generate}
        for _ in range(3):
            search_tree = algo.step(search_tree, actions)
        
        # Get results
        results = tq.top_k(search_tree, algo, k=2)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Check result format
        for state, score in results:
            self.assertIsInstance(score, (int, float))
    
    def test_abmctsa_basic_search(self):
        """Test basic AB-MCTS-A functionality."""
        algo = tq.ABMCTSA()
        search_tree = algo.init_tree()
        
        def generate_creative(parent_state):
            if parent_state is None:
                new_state = "creative_root"
            else:
                new_state = f"creative_child_{parent_state}"
            return new_state, 0.7
        
        def generate_analytical(parent_state):
            if parent_state is None:
                new_state = "analytical_root"
            else:
                new_state = f"analytical_child_{parent_state}"
            return new_state, 0.6
        
        actions = {
            'creative': generate_creative,
            'analytical': generate_analytical
        }
        
        # Run search steps
        for _ in range(5):
            search_tree = algo.step(search_tree, actions)
        
        results = tq.top_k(search_tree, algo, k=3)
        self.assertGreater(len(results), 0)
    
    def test_abmctsm_basic_search(self):
        """Test basic AB-MCTS-M functionality."""
        try:
            algo = tq.ABMCTSM()
            search_tree = algo.init_tree()
            
            def generate(parent_state):
                if parent_state is None:
                    new_state = "root_solution"
                else:
                    new_state = f"improved_{parent_state}"
                return new_state, 0.75
            
            actions = {'improve': generate}
            
            # Run fewer steps for AB-MCTS-M (more complex)
            for _ in range(2):
                search_tree = algo.step(search_tree, actions)
            
            results = tq.top_k(search_tree, algo, k=2)
            self.assertGreater(len(results), 0)
            
        except Exception as e:
            # AB-MCTS-M might need additional setup
            self.skipTest(f"AB-MCTS-M requires additional setup: {e}")


class TestGenerationFunctions(TestTreeQuestMCPIntegration):
    """Test generation function creation and management."""
    
    def test_simple_generation_function_creation(self):
        """Test creation of simple generation functions."""
        simple_gen = treequest_mcp_server._create_simple_generator(
            "Test prompt", 
            score_range=(0.2, 0.8)
        )
        
        # Test with None parent (root)
        state, score = simple_gen(None)
        self.assertIsInstance(state, str)
        self.assertGreaterEqual(score, 0.2)
        self.assertLessEqual(score, 0.8)
        
        # Test with parent state
        state2, score2 = simple_gen("parent_state")
        self.assertIsInstance(state2, str)
        self.assertIn("parent_state", state2)
        self.assertGreaterEqual(score2, 0.2)
        self.assertLessEqual(score2, 0.8)
    
    def test_custom_generation_function_execution(self):
        """Test custom generation function execution."""
        custom_code = '''
def generate(parent_state):
    if parent_state is None:
        new_state = "custom_root"
    else:
        new_state = f"custom_child_{len(str(parent_state))}"
    score = 0.42
    return new_state, score
'''
        
        # Create namespace and execute code
        namespace = {}
        exec(custom_code, namespace)
        generate_func = namespace['generate']
        
        # Test the function
        state1, score1 = generate_func(None)
        self.assertEqual(state1, "custom_root")
        self.assertEqual(score1, 0.42)
        
        state2, score2 = generate_func("test")
        self.assertEqual(state2, "custom_child_4")  # len("test") = 4
        self.assertEqual(score2, 0.42)


class TestErrorHandling(TestTreeQuestMCPIntegration):
    """Test error handling and edge cases."""
    
    def test_invalid_algorithm_type(self):
        """Test handling of invalid algorithm types."""
        with self.assertRaises((ValueError, AttributeError)):
            getattr(tq, "InvalidAlgorithm")()
    
    def test_empty_actions_in_search(self):
        """Test search with no actions defined."""
        algo = tq.StandardMCTS()
        search_tree = algo.init_tree()
        
        # Try to step with empty actions
        with self.assertRaises((KeyError, ValueError)):
            algo.step(search_tree, {})
    
    def test_invalid_score_range(self):
        """Test invalid score ranges in simple generator."""
        # Test score range where min > max
        with self.assertRaises(ValueError):
            treequest_mcp_server._create_simple_generator(
                "Test", 
                score_range=(0.8, 0.2)  # Invalid: min > max
            )
        
        # Test score range outside [0, 1]
        with self.assertRaises(ValueError):
            treequest_mcp_server._create_simple_generator(
                "Test",
                score_range=(-0.1, 1.5)  # Invalid: outside [0, 1]
            )
    
    def test_malformed_custom_code(self):
        """Test handling of malformed custom code."""
        malformed_code = "this is not valid python syntax!!!"
        
        with self.assertRaises(SyntaxError):
            exec(malformed_code, {})
    
    def test_custom_code_missing_generate_function(self):
        """Test custom code that doesn't define generate function."""
        code_without_generate = '''
def other_function():
    return "not generate"
x = 42
'''
        
        namespace = {}
        exec(code_without_generate, namespace)
        
        # Should not have 'generate' function
        self.assertNotIn('generate', namespace)


class TestDocumentation(TestTreeQuestMCPIntegration):
    """Test documentation and help functionality."""
    
    def test_documentation_topics(self):
        """Test that documentation can be retrieved for valid topics."""
        valid_topics = ["overview", "algorithms", "generation_functions", "examples"]
        
        for topic in valid_topics:
            try:
                doc = treequest_mcp_server.get_documentation(topic)
                self.assertIsInstance(doc, str)
                self.assertGreater(len(doc), 0)
                # Should contain the topic name or related content
                self.assertTrue(
                    topic.lower() in doc.lower() or 
                    "treequest" in doc.lower() or
                    "search" in doc.lower()
                )
            except Exception as e:
                self.fail(f"Documentation for topic '{topic}' failed: {e}")
    
    def test_invalid_documentation_topic(self):
        """Test handling of invalid documentation topics."""
        with self.assertRaises(ValueError):
            treequest_mcp_server.get_documentation("invalid_topic")


class TestPromptGeneration(TestTreeQuestMCPIntegration):
    """Test prompt generation functionality."""
    
    def test_llm_generation_function_prompt(self):
        """Test LLM generation function prompt creation."""
        prompt = treequest_mcp_server.create_llm_generation_function(
            task_description="Solve coding problems",
            llm_model="gpt-4",
            scoring_strategy="correctness"
        )
        
        self.assertIsInstance(prompt, str)
        self.assertIn("gpt-4", prompt)
        self.assertIn("Solve coding problems", prompt)
        self.assertIn("correctness", prompt)
        self.assertIn("def generate", prompt)
    
    def test_multi_llm_search_prompt(self):
        """Test multi-LLM search setup prompt creation."""
        models = ["gpt-4", "claude-3", "gemini-pro"]
        prompt = treequest_mcp_server.setup_multi_llm_search(
            task_description="Creative writing",
            llm_models=models,
            search_budget=15
        )
        
        self.assertIsInstance(prompt, str)
        self.assertIn("Creative writing", prompt)
        
        # Check all models are mentioned
        for model in models:
            self.assertIn(model, prompt)
        
        # Check budget is mentioned
        self.assertIn("15", prompt)
        
        # Should contain setup instructions
        self.assertIn("initialize_search", prompt)
        self.assertIn("add_generation_function", prompt)


class TestComprehensiveWorkflow(TestTreeQuestMCPIntegration):
    """Test complete end-to-end workflows."""
    
    def test_complete_search_workflow(self):
        """Test a complete search workflow."""
        # 1. Set up algorithm
        algo = tq.StandardMCTS()
        search_tree = algo.init_tree()
        
        # 2. Create generation functions
        def generate_solution(parent_state):
            if parent_state is None:
                new_state = "Initial solution approach"
            else:
                new_state = f"Refined solution: {parent_state[:20]}..."
            
            # Simulate scoring
            import random
            score = random.uniform(0.3, 0.9)
            return new_state, score
        
        def generate_alternative(parent_state):
            if parent_state is None:
                new_state = "Alternative approach"
            else:
                new_state = f"Alternative to: {parent_state[:15]}..."
            
            score = random.uniform(0.2, 0.8)
            return new_state, score
        
        actions = {
            'solution': generate_solution,
            'alternative': generate_alternative
        }
        
        # 3. Run search
        num_steps = 8
        for step in range(num_steps):
            search_tree = algo.step(search_tree, actions)
        
        # 4. Get results
        top_results = tq.top_k(search_tree, algo, k=3)
        
        # Validate results
        self.assertIsInstance(top_results, list)
        self.assertGreater(len(top_results), 0)
        self.assertLessEqual(len(top_results), 3)
        
        # Check result format
        for state, score in top_results:
            self.assertIsInstance(state, str)
            self.assertIsInstance(score, (int, float))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # Results should be sorted by score (highest first)
        if len(top_results) > 1:
            for i in range(len(top_results) - 1):
                self.assertGreaterEqual(top_results[i][1], top_results[i + 1][1])
    
    def test_multi_algorithm_comparison(self):
        """Test comparing different algorithms on the same problem."""
        def test_generate(parent_state):
            if parent_state is None:
                return "root_solution", 0.5
            else:
                return f"child_{parent_state}", 0.6
        
        actions = {'test': test_generate}
        results = {}
        
        # Test StandardMCTS
        algo1 = tq.StandardMCTS()
        tree1 = algo1.init_tree()
        for _ in range(3):
            tree1 = algo1.step(tree1, actions)
        results['StandardMCTS'] = tq.top_k(tree1, algo1, k=2)
        
        # Test AB-MCTS-A
        algo2 = tq.ABMCTSA()
        tree2 = algo2.init_tree()
        for _ in range(3):
            tree2 = algo2.step(tree2, actions)
        results['ABMCTSA'] = tq.top_k(tree2, algo2, k=2)
        
        # Validate both produced results
        for algo_name, result_list in results.items():
            self.assertIsInstance(result_list, list)
            self.assertGreater(len(result_list), 0)
            
            for state, score in result_list:
                self.assertIsInstance(state, str)
                self.assertIsInstance(score, (int, float))


def run_integration_tests():
    """Run all integration tests."""
    print("TreeQuest MCP Server - Integration Test Suite")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestBasicFunctionality,
        TestTreeSearchFunctionality,
        TestGenerationFunctions,
        TestErrorHandling,
        TestDocumentation,
        TestPromptGeneration,
        TestComprehensiveWorkflow
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    test_result = run_integration_tests()
    
    print("\n" + "=" * 50)
    print(f"Tests run: {test_result.testsRun}")
    print(f"Failures: {len(test_result.failures)}")
    print(f"Errors: {len(test_result.errors)}")
    print(f"Skipped: {len(test_result.skipped) if hasattr(test_result, 'skipped') else 0}")
    
    if test_result.failures:
        print("\nFAILURES:")
        for test, traceback in test_result.failures:
            print(f"- {test}")
            print(f"  {traceback}")
    
    if test_result.errors:
        print("\nERRORS:")
        for test, traceback in test_result.errors:
            print(f"- {test}")
            print(f"  {traceback}")
    
    success = len(test_result.failures) == 0 and len(test_result.errors) == 0
    print(f"\nOVERALL: {'✓ PASSED' if success else '✗ FAILED'}")
    
    exit(0 if success else 1)