#!/usr/bin/env python3
"""
Comprehensive Unit Tests for TreeQuest MCP Server

This test suite covers all MCP server functionality including:
- All tools (initialize_search, add_generation_function, run_search_steps, etc.)
- All resources (documentation, session info)
- All prompts (LLM generation functions, multi-LLM setup)
- Error handling and edge cases
- Session management and persistence
"""

import unittest
import sys
import json
import tempfile
import os
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the MCP server components
from treequest_mcp_server import (
    mcp, _search_states, _generation_functions,
    initialize_search, add_generation_function, 
    run_search_steps, get_top_results,
    list_sessions, save_session,
    get_documentation, get_session_info,
    create_llm_generation_function, setup_multi_llm_search
)

import treequest as tq


class TestTreeQuestMCPServer(unittest.TestCase):
    """Test suite for TreeQuest MCP Server functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Clear global state before each test
        _search_states.clear()
        _generation_functions.clear()
        
        # Test session data
        self.test_session_id = "test_session"
        self.test_algorithm = "StandardMCTS"
        self.test_action_name = "test_action"
    
    def tearDown(self):
        """Clean up after each test method."""
        # Clear global state after each test
        _search_states.clear()
        _generation_functions.clear()


class TestSearchInitialization(TestTreeQuestMCPServer):
    """Test search session initialization functionality."""
    
    def test_initialize_search_standard_mcts(self):
        """Test initializing a StandardMCTS search session."""
        result = initialize_search_impl(
            algorithm_type="StandardMCTS",
            session_id=self.test_session_id
        )
        
        self.assertIn("successfully initialized", result)
        self.assertIn(self.test_session_id, _search_states)
        self.assertEqual(_search_states[self.test_session_id]["algorithm_type"], "StandardMCTS")
    
    def test_initialize_search_abmctsa(self):
        """Test initializing an AB-MCTS-A search session."""
        result = initialize_search_impl(
            algorithm_type="ABMCTSA",
            session_id=self.test_session_id
        )
        
        self.assertIn("successfully initialized", result)
        self.assertIn(self.test_session_id, _search_states)
        self.assertEqual(_search_states[self.test_session_id]["algorithm_type"], "ABMCTSA")
    
    def test_initialize_search_abmctsm(self):
        """Test initializing an AB-MCTS-M search session."""
        result = initialize_search_impl(
            algorithm_type="ABMCTSM",
            session_id=self.test_session_id
        )
        
        self.assertIn("successfully initialized", result)
        self.assertIn(self.test_session_id, _search_states)
        self.assertEqual(_search_states[self.test_session_id]["algorithm_type"], "ABMCTSM")
    
    def test_initialize_search_tree_of_thoughts(self):
        """Test initializing a TreeOfThoughts search session."""
        result = initialize_search_impl(
            algorithm_type="TreeOfThoughtsBFS",
            session_id=self.test_session_id
        )
        
        self.assertIn("successfully initialized", result)
        self.assertIn(self.test_session_id, _search_states)
        self.assertEqual(_search_states[self.test_session_id]["algorithm_type"], "TreeOfThoughtsBFS")
    
    def test_initialize_search_with_params(self):
        """Test initializing search with custom algorithm parameters."""
        params = {"exploration_weight": 1.5, "samples_per_action": 3}
        result = initialize_search_impl(
            algorithm_type="StandardMCTS",
            session_id=self.test_session_id,
            algorithm_params=params
        )
        
        self.assertIn("successfully initialized", result)
        self.assertEqual(_search_states[self.test_session_id]["algorithm_params"], params)
    
    def test_initialize_search_invalid_algorithm(self):
        """Test error handling for invalid algorithm type."""
        with self.assertRaises(ValueError):
            initialize_search_impl(
                algorithm_type="InvalidAlgorithm",
                session_id=self.test_session_id
            )
    
    def test_initialize_search_duplicate_session(self):
        """Test handling of duplicate session IDs."""
        # Initialize first session
        initialize_search_impl("StandardMCTS", self.test_session_id)
        
        # Initialize with same session ID should replace
        result = initialize_search_impl("ABMCTSA", self.test_session_id)
        
        self.assertIn("successfully initialized", result)
        self.assertEqual(_search_states[self.test_session_id]["algorithm_type"], "ABMCTSA")


class TestGenerationFunctions(TestTreeQuestMCPServer):
    """Test generation function management."""
    
    def setUp(self):
        """Set up search session for generation function tests."""
        super().setUp()
        initialize_search_impl("StandardMCTS", self.test_session_id)
    
    def test_add_simple_generation_function(self):
        """Test adding a simple generation function."""
        result = add_generation_function_impl(
            session_id=self.test_session_id,
            action_name=self.test_action_name,
            function_type="simple",
            prompt="Test prompt",
            score_range=(0.0, 1.0)
        )
        
        self.assertIn("successfully added", result)
        self.assertIn(self.test_session_id, _generation_functions)
        self.assertIn(self.test_action_name, _generation_functions[self.test_session_id])
    
    def test_add_custom_generation_function(self):
        """Test adding a custom generation function."""
        custom_code = '''
def generate(parent_state):
    if parent_state is None:
        new_state = "root"
    else:
        new_state = f"child_of_{parent_state}"
    score = 0.8
    return new_state, score
'''
        
        result = add_generation_function_impl(
            session_id=self.test_session_id,
            action_name=self.test_action_name,
            function_type="custom",
            custom_code=custom_code
        )
        
        self.assertIn("successfully added", result)
        self.assertIn(self.test_session_id, _generation_functions)
        
        # Test that the function is callable
        func_info = _generation_functions[self.test_session_id][self.test_action_name]
        self.assertIn("function", func_info)
        
        # Test the function works
        state, score = func_info["function"](None)
        self.assertEqual(state, "root")
        self.assertEqual(score, 0.8)
    
    def test_add_generation_function_invalid_session(self):
        """Test error handling for invalid session ID."""
        with self.assertRaises(ValueError):
            add_generation_function_impl(
                session_id="nonexistent_session",
                action_name=self.test_action_name,
                function_type="simple",
                prompt="Test prompt"
            )
    
    def test_add_generation_function_invalid_type(self):
        """Test error handling for invalid function type."""
        with self.assertRaises(ValueError):
            add_generation_function_impl(
                session_id=self.test_session_id,
                action_name=self.test_action_name,
                function_type="invalid_type"
            )
    
    def test_add_generation_function_invalid_custom_code(self):
        """Test error handling for invalid custom code."""
        invalid_code = "this is not valid python code!!!"
        
        with self.assertRaises(Exception):
            add_generation_function_impl(
                session_id=self.test_session_id,
                action_name=self.test_action_name,
                function_type="custom",
                custom_code=invalid_code
            )
    
    def test_add_generation_function_missing_generate_function(self):
        """Test error handling when custom code doesn't define generate function."""
        invalid_code = '''
def other_function():
    return "not generate"
'''
        
        with self.assertRaises(ValueError):
            add_generation_function_impl(
                session_id=self.test_session_id,
                action_name=self.test_action_name,
                function_type="custom",
                custom_code=invalid_code
            )
    
    def test_multiple_generation_functions(self):
        """Test adding multiple generation functions to same session."""
        # Add first function
        add_generation_function_impl(
            session_id=self.test_session_id,
            action_name="action1",
            function_type="simple",
            prompt="First prompt",
            score_range=(0.0, 1.0)
        )
        
        # Add second function
        add_generation_function_impl(
            session_id=self.test_session_id,
            action_name="action2",
            function_type="simple",
            prompt="Second prompt",
            score_range=(0.2, 0.8)
        )
        
        # Check both functions exist
        self.assertIn("action1", _generation_functions[self.test_session_id])
        self.assertIn("action2", _generation_functions[self.test_session_id])
        self.assertEqual(len(_generation_functions[self.test_session_id]), 2)


class TestSearchExecution(TestTreeQuestMCPServer):
    """Test search execution functionality."""
    
    def setUp(self):
        """Set up search session with generation function for execution tests."""
        super().setUp()
        initialize_search_impl("StandardMCTS", self.test_session_id)
        add_generation_function_impl(
            session_id=self.test_session_id,
            action_name=self.test_action_name,
            function_type="simple",
            prompt="Test prompt",
            score_range=(0.0, 1.0)
        )
    
    def test_run_search_steps_basic(self):
        """Test running basic search steps."""
        result = run_search_steps_impl(
            session_id=self.test_session_id,
            num_steps=3,
            log_progress=False
        )
        
        self.assertIn("Completed 3 search steps", result)
        
        # Check that search tree has been updated
        search_state = _search_states[self.test_session_id]
        self.assertIsNotNone(search_state["search_tree"])
        self.assertGreater(search_state["total_steps"], 0)
    
    def test_run_search_steps_with_logging(self):
        """Test running search steps with progress logging."""
        result = run_search_steps_impl(
            session_id=self.test_session_id,
            num_steps=2,
            log_progress=True
        )
        
        self.assertIn("Completed 2 search steps", result)
    
    def test_run_search_steps_invalid_session(self):
        """Test error handling for invalid session ID."""
        with self.assertRaises(ValueError):
            run_search_steps_impl(
                session_id="nonexistent_session",
                num_steps=3
            )
    
    def test_run_search_steps_no_generation_functions(self):
        """Test error handling when no generation functions are defined."""
        # Create session without generation functions
        initialize_search_impl("StandardMCTS", "empty_session")
        
        with self.assertRaises(ValueError):
            run_search_steps_impl(
                session_id="empty_session",
                num_steps=3
            )
    
    def test_run_search_steps_zero_steps(self):
        """Test handling of zero search steps."""
        result = run_search_steps_impl(
            session_id=self.test_session_id,
            num_steps=0
        )
        
        self.assertIn("Completed 0 search steps", result)
    
    def test_run_search_steps_negative_steps(self):
        """Test error handling for negative number of steps."""
        with self.assertRaises(ValueError):
            run_search_steps_impl(
                session_id=self.test_session_id,
                num_steps=-1
            )


class TestResultRetrieval(TestTreeQuestMCPServer):
    """Test result retrieval functionality."""
    
    def setUp(self):
        """Set up search session and run some steps for result tests."""
        super().setUp()
        initialize_search_impl("StandardMCTS", self.test_session_id)
        add_generation_function_impl(
            session_id=self.test_session_id,
            action_name=self.test_action_name,
            function_type="simple",
            prompt="Test prompt",
            score_range=(0.0, 1.0)
        )
        # Run some search steps to generate results
        run_search_steps_impl(self.test_session_id, 5)
    
    def test_get_top_results_basic(self):
        """Test getting top results from search."""
        result = get_top_results_impl(
            session_id=self.test_session_id,
            k=3
        )
        
        self.assertIn("Top 3 results", result)
        self.assertIn("State:", result)
        self.assertIn("Score:", result)
    
    def test_get_top_results_large_k(self):
        """Test getting more results than available."""
        result = get_top_results_impl(
            session_id=self.test_session_id,
            k=100  # More than we have
        )
        
        self.assertIn("results", result)
    
    def test_get_top_results_invalid_session(self):
        """Test error handling for invalid session ID."""
        with self.assertRaises(ValueError):
            get_top_results_impl(
                session_id="nonexistent_session",
                k=3
            )
    
    def test_get_top_results_zero_k(self):
        """Test handling of k=0."""
        result = get_top_results_impl(
            session_id=self.test_session_id,
            k=0
        )
        
        self.assertIn("Top 0 results", result)
    
    def test_get_top_results_negative_k(self):
        """Test error handling for negative k."""
        with self.assertRaises(ValueError):
            get_top_results_impl(
                session_id=self.test_session_id,
                k=-1
            )


class TestSessionManagement(TestTreeQuestMCPServer):
    """Test session management functionality."""
    
    def test_list_sessions_empty(self):
        """Test listing sessions when none exist."""
        result = list_sessions_impl()
        
        self.assertIn("No active sessions", result)
    
    def test_list_sessions_with_sessions(self):
        """Test listing sessions when multiple exist."""
        # Create multiple sessions
        initialize_search_impl("StandardMCTS", "session1")
        initialize_search_impl("ABMCTSA", "session2")
        initialize_search_impl("ABMCTSM", "session3")
        
        result = list_sessions_impl()
        
        self.assertIn("Active sessions (3)", result)
        self.assertIn("session1", result)
        self.assertIn("session2", result)
        self.assertIn("session3", result)
        self.assertIn("StandardMCTS", result)
        self.assertIn("ABMCTSA", result)
        self.assertIn("ABMCTSM", result)
    
    def test_save_session_basic(self):
        """Test saving a session to file."""
        # Set up session with some data
        initialize_search_impl("StandardMCTS", self.test_session_id)
        add_generation_function_impl(
            session_id=self.test_session_id,
            action_name=self.test_action_name,
            function_type="simple",
            prompt="Test prompt"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_session.pkl")
            
            result = save_session_impl(
                session_id=self.test_session_id,
                filepath=filepath
            )
            
            self.assertIn("saved successfully", result)
            self.assertTrue(os.path.exists(filepath))
            
            # Verify file contents
            with open(filepath, 'rb') as f:
                saved_data = pickle.load(f)
            
            self.assertIn("search_state", saved_data)
            self.assertIn("generation_functions", saved_data)
    
    def test_save_session_default_filepath(self):
        """Test saving session with default filepath."""
        initialize_search_impl("StandardMCTS", self.test_session_id)
        
        result = save_session_impl(session_id=self.test_session_id)
        
        self.assertIn("saved successfully", result)
        # Clean up the created file
        default_path = f"{self.test_session_id}.pkl"
        if os.path.exists(default_path):
            os.remove(default_path)
    
    def test_save_session_invalid_session(self):
        """Test error handling for saving nonexistent session."""
        with self.assertRaises(ValueError):
            save_session_impl(session_id="nonexistent_session")


class TestResources(TestTreeQuestMCPServer):
    """Test MCP resource functionality."""
    
    def test_get_docs_overview(self):
        """Test getting overview documentation."""
        result = get_docs_resource("overview")
        
        self.assertIn("TreeQuest", result)
        self.assertIn("tree search", result)
        self.assertIn("AB-MCTS", result)
    
    def test_get_docs_algorithms(self):
        """Test getting algorithm documentation."""
        result = get_docs_resource("algorithms")
        
        self.assertIn("StandardMCTS", result)
        self.assertIn("ABMCTSA", result)
        self.assertIn("ABMCTSM", result)
        self.assertIn("TreeOfThoughtsBFS", result)
    
    def test_get_docs_generation_functions(self):
        """Test getting generation function documentation."""
        result = get_docs_resource("generation_functions")
        
        self.assertIn("generation function", result)
        self.assertIn("generate", result)
        self.assertIn("parent_state", result)
    
    def test_get_docs_examples(self):
        """Test getting examples documentation."""
        result = get_docs_resource("examples")
        
        self.assertIn("example", result)
        self.assertIn("initialize_search", result)
    
    def test_get_docs_invalid_topic(self):
        """Test error handling for invalid documentation topic."""
        with self.assertRaises(ValueError):
            get_docs_resource("invalid_topic")
    
    def test_get_session_info_valid(self):
        """Test getting session information for valid session."""
        # Set up session with data
        initialize_search_impl("StandardMCTS", self.test_session_id)
        add_generation_function_impl(
            session_id=self.test_session_id,
            action_name=self.test_action_name,
            function_type="simple",
            prompt="Test prompt"
        )
        run_search_steps_impl(self.test_session_id, 3)
        
        result = get_session_info_resource(self.test_session_id)
        
        self.assertIn("Session Information", result)
        self.assertIn(self.test_session_id, result)
        self.assertIn("StandardMCTS", result)
        self.assertIn("Generation Functions", result)
        self.assertIn(self.test_action_name, result)
    
    def test_get_session_info_invalid(self):
        """Test error handling for invalid session ID in resource."""
        with self.assertRaises(ValueError):
            get_session_info_resource("nonexistent_session")


class TestPrompts(TestTreeQuestMCPServer):
    """Test MCP prompt functionality."""
    
    def test_create_llm_generation_function_prompt(self):
        """Test creating LLM generation function prompt."""
        result = create_llm_generation_function_prompt(
            task_description="Solve math problems",
            llm_model="gpt-4",
            scoring_strategy="quality"
        )
        
        self.assertIn("gpt-4", result)
        self.assertIn("Solve math problems", result)
        self.assertIn("quality", result)
        self.assertIn("def generate", result)
        self.assertIn("parent_state", result)
    
    def test_setup_multi_llm_search_prompt(self):
        """Test creating multi-LLM search setup prompt."""
        models = ["gpt-4", "claude-3", "gemini-pro"]
        result = setup_multi_llm_search_prompt(
            task_description="Write creative stories",
            llm_models=models,
            search_budget=20
        )
        
        self.assertIn("Write creative stories", result)
        self.assertIn("gpt-4", result)
        self.assertIn("claude-3", result)
        self.assertIn("gemini-pro", result)
        self.assertIn("20", result)
        self.assertIn("initialize_search", result)
        self.assertIn("add_generation_function", result)


class TestErrorHandling(TestTreeQuestMCPServer):
    """Test comprehensive error handling."""
    
    def test_empty_session_id(self):
        """Test handling of empty session ID."""
        with self.assertRaises(ValueError):
            initialize_search_impl("StandardMCTS", "")
    
    def test_none_session_id(self):
        """Test handling of None session ID."""
        with self.assertRaises((ValueError, TypeError)):
            initialize_search_impl("StandardMCTS", None)
    
    def test_empty_action_name(self):
        """Test handling of empty action name."""
        initialize_search_impl("StandardMCTS", self.test_session_id)
        
        with self.assertRaises(ValueError):
            add_generation_function_impl(
                session_id=self.test_session_id,
                action_name="",
                function_type="simple",
                prompt="Test"
            )
    
    def test_invalid_score_range(self):
        """Test handling of invalid score ranges."""
        initialize_search_impl("StandardMCTS", self.test_session_id)
        
        # Test invalid range (min > max)
        with self.assertRaises(ValueError):
            add_generation_function_impl(
                session_id=self.test_session_id,
                action_name=self.test_action_name,
                function_type="simple",
                prompt="Test",
                score_range=(1.0, 0.0)
            )
        
        # Test range outside [0, 1]
        with self.assertRaises(ValueError):
            add_generation_function_impl(
                session_id=self.test_session_id,
                action_name=self.test_action_name,
                function_type="simple",
                prompt="Test",
                score_range=(-1.0, 2.0)
            )


class TestIntegration(TestTreeQuestMCPServer):
    """Integration tests for complete workflows."""
    
    def test_complete_search_workflow(self):
        """Test complete workflow from initialization to results."""
        # 1. Initialize search
        result = initialize_search_impl("StandardMCTS", self.test_session_id)
        self.assertIn("successfully initialized", result)
        
        # 2. Add generation function
        result = add_generation_function_impl(
            session_id=self.test_session_id,
            action_name="explore",
            function_type="simple",
            prompt="Explore the problem space",
            score_range=(0.0, 1.0)
        )
        self.assertIn("successfully added", result)
        
        # 3. Run search steps
        result = run_search_steps_impl(self.test_session_id, 5)
        self.assertIn("Completed 5 search steps", result)
        
        # 4. Get results
        result = get_top_results_impl(self.test_session_id, 3)
        self.assertIn("Top 3 results", result)
        
        # 5. Save session
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "integration_test.pkl")
            result = save_session_impl(self.test_session_id, filepath)
            self.assertIn("saved successfully", result)
            self.assertTrue(os.path.exists(filepath))
    
    def test_multi_action_workflow(self):
        """Test workflow with multiple generation functions."""
        # Initialize search
        initialize_search_impl("ABMCTSA", self.test_session_id)
        
        # Add multiple generation functions
        add_generation_function_impl(
            session_id=self.test_session_id,
            action_name="creative",
            function_type="simple",
            prompt="Be creative",
            score_range=(0.4, 1.0)
        )
        
        add_generation_function_impl(
            session_id=self.test_session_id,
            action_name="analytical",
            function_type="simple",
            prompt="Be analytical",
            score_range=(0.3, 0.9)
        )
        
        # Run search with multiple actions
        result = run_search_steps_impl(self.test_session_id, 10)
        self.assertIn("Completed 10 search steps", result)
        
        # Verify session has both functions
        self.assertEqual(len(_generation_functions[self.test_session_id]), 2)
        self.assertIn("creative", _generation_functions[self.test_session_id])
        self.assertIn("analytical", _generation_functions[self.test_session_id])
    
    def test_custom_function_workflow(self):
        """Test workflow with custom generation function."""
        initialize_search_impl("StandardMCTS", self.test_session_id)
        
        custom_code = '''
def generate(parent_state):
    import random
    if parent_state is None:
        new_state = "Initial solution"
    else:
        new_state = f"Improved: {parent_state}"
    
    score = random.uniform(0.2, 0.9)
    return new_state, score
'''
        
        add_generation_function_impl(
            session_id=self.test_session_id,
            action_name="custom_solver",
            function_type="custom",
            custom_code=custom_code
        )
        
        run_search_steps_impl(self.test_session_id, 3)
        result = get_top_results_impl(self.test_session_id, 2)
        
        self.assertIn("Initial solution", result)


def run_tests():
    """Run all unit tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestSearchInitialization,
        TestGenerationFunctions,
        TestSearchExecution,
        TestResultRetrieval,
        TestSessionManagement,
        TestResources,
        TestPrompts,
        TestErrorHandling,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("TreeQuest MCP Server - Comprehensive Unit Test Suite")
    print("=" * 60)
    
    # Run tests
    test_result = run_tests()
    
    print("\n" + "=" * 60)
    print(f"Tests run: {test_result.testsRun}")
    print(f"Failures: {len(test_result.failures)}")
    print(f"Errors: {len(test_result.errors)}")
    print(f"Skipped: {len(test_result.skipped) if hasattr(test_result, 'skipped') else 0}")
    
    if test_result.failures:
        print("\nFAILURES:")
        for test, traceback in test_result.failures:
            print(f"- {test}: {traceback}")
    
    if test_result.errors:
        print("\nERRORS:")
        for test, traceback in test_result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(test_result.failures) == 0 and len(test_result.errors) == 0
    print(f"\nOVERALL: {'✓ PASSED' if success else '✗ FAILED'}")
    
    exit(0 if success else 1)