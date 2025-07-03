"""
TreeQuest MCP Server

A Model Context Protocol server that exposes TreeQuest's tree search algorithms
for LLM inference-time scaling. Provides tools for running AB-MCTS and other
tree search algorithms with customizable generation functions.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import tempfile
import base64
from dataclasses import dataclass, asdict
import traceback

from fastmcp import FastMCP

# Add the src directory to Python path to import treequest
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import treequest as tq
    from treequest.types import GenerateFnType, StateScoreType
    TREEQUEST_AVAILABLE = True
except ImportError as e:
    TREEQUEST_AVAILABLE = False
    IMPORT_ERROR = str(e)

# MCP Server setup
mcp = FastMCP(
    "TreeQuest",
    dependencies=["scipy>=1.0.0"] + (["fastmcp>=2.0.0"] if TREEQUEST_AVAILABLE else [])
)

# Global storage for search states and generation functions
_search_states: Dict[str, Any] = {}
_generation_functions: Dict[str, Dict[str, Any]] = {}


@dataclass
class SearchConfig:
    """Configuration for a tree search session."""
    algorithm_type: str
    session_id: str
    algorithm_params: Dict[str, Any]
    current_step: int = 0
    total_nodes_generated: int = 0


@dataclass 
class GenerationResult:
    """Result from a generation function."""
    state: Any
    score: float
    metadata: Optional[Dict[str, Any]] = None


def _check_treequest_available():
    """Check if TreeQuest is available and raise error if not."""
    if not TREEQUEST_AVAILABLE:
        raise RuntimeError(f"TreeQuest is not available. Import error: {IMPORT_ERROR}")


def _create_simple_generator(prompt: str, score_range: Tuple[float, float] = (0.0, 1.0)):
    """Create a simple generation function for demonstration purposes."""
    import random
    
    def generate(parent_state: Optional[str] = None) -> Tuple[str, float]:
        if parent_state is None:
            new_state = f"Initial response to: {prompt}"
        else:
            new_state = f"Refined response based on: {parent_state[:100]}..."
        
        score = random.uniform(score_range[0], score_range[1])
        return new_state, score
    
    return generate


@mcp.tool()
def initialize_search(
    algorithm_type: str,
    session_id: str = "default",
    algorithm_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Initialize a new tree search session with the specified algorithm.
    
    Args:
        algorithm_type: Type of algorithm to use ('ABMCTSA', 'ABMCTSM', 'StandardMCTS', 'TreeOfThoughtsBFS')
        session_id: Unique identifier for this search session
        algorithm_params: Optional parameters for the algorithm (e.g., exploration_weight, samples_per_action)
    
    Returns:
        JSON string with initialization status and session info
    """
    _check_treequest_available()
    
    if algorithm_params is None:
        algorithm_params = {}
    
    try:
        # Create the algorithm instance
        if algorithm_type == "ABMCTSA":
            algo = tq.ABMCTSA(**algorithm_params)
        elif algorithm_type == "ABMCTSM": 
            algo = tq.ABMCTSM(**algorithm_params)
        elif algorithm_type == "StandardMCTS":
            algo = tq.StandardMCTS(**algorithm_params)
        elif algorithm_type == "TreeOfThoughtsBFS":
            algo = tq.TreeOfThoughtsBFSAlgo(**algorithm_params)
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
        
        # Initialize the search tree
        search_tree = algo.init_tree()
        
        # Store the state
        _search_states[session_id] = {
            "algorithm": algo,
            "search_tree": search_tree,
            "config": SearchConfig(algorithm_type, session_id, algorithm_params)
        }
        
        _generation_functions[session_id] = {}
        
        result = {
            "status": "success",
            "session_id": session_id,
            "algorithm_type": algorithm_type,
            "algorithm_params": algorithm_params,
            "message": f"Initialized {algorithm_type} search session '{session_id}'"
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "session_id": session_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        return json.dumps(error_result, indent=2)


@mcp.tool()
def add_generation_function(
    session_id: str,
    action_name: str,
    function_type: str = "simple",
    prompt: Optional[str] = None,
    score_range: Tuple[float, float] = (0.0, 1.0),
    custom_code: Optional[str] = None
) -> str:
    """
    Add a generation function to a search session.
    
    Args:
        session_id: The search session ID
        action_name: Name/label for this action type
        function_type: Type of function ('simple' for demo, 'custom' for user-defined code)
        prompt: Prompt text for simple function type
        score_range: Score range for simple function type (min, max)
        custom_code: Python code for custom function (must define a 'generate' function)
    
    Returns:
        JSON string with operation status
    """
    _check_treequest_available()
    
    if session_id not in _search_states:
        return json.dumps({
            "status": "error",
            "error": f"Session '{session_id}' not found. Initialize a search session first."
        })
    
    try:
        if function_type == "simple":
            if prompt is None:
                prompt = f"Generate response for action '{action_name}'"
            gen_fn = _create_simple_generator(prompt, score_range)
            
        elif function_type == "custom":
            if custom_code is None:
                raise ValueError("custom_code is required for custom function type")
            
            # Execute the custom code in a controlled environment
            namespace = {"Optional": Optional, "Tuple": Tuple}
            exec(custom_code, namespace)
            
            if "generate" not in namespace:
                raise ValueError("Custom code must define a 'generate' function")
            
            gen_fn = namespace["generate"]
            
        else:
            raise ValueError(f"Unknown function_type: {function_type}")
        
        # Store the generation function
        _generation_functions[session_id][action_name] = gen_fn
        
        result = {
            "status": "success",
            "session_id": session_id,
            "action_name": action_name,
            "function_type": function_type,
            "message": f"Added generation function '{action_name}' to session '{session_id}'"
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "session_id": session_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        return json.dumps(error_result, indent=2)


@mcp.tool()
def run_search_steps(
    session_id: str,
    num_steps: int,
    log_progress: bool = True
) -> str:
    """
    Run multiple steps of the tree search algorithm.
    
    Args:
        session_id: The search session ID
        num_steps: Number of search steps to execute
        log_progress: Whether to log progress during search
    
    Returns:
        JSON string with search results and statistics
    """
    _check_treequest_available()
    
    if session_id not in _search_states:
        return json.dumps({
            "status": "error",
            "error": f"Session '{session_id}' not found. Initialize a search session first."
        })
    
    if session_id not in _generation_functions or not _generation_functions[session_id]:
        return json.dumps({
            "status": "error", 
            "error": f"No generation functions defined for session '{session_id}'. Add generation functions first."
        })
    
    try:
        state_data = _search_states[session_id]
        algo = state_data["algorithm"]
        search_tree = state_data["search_tree"]
        config = state_data["config"]
        generate_fns = _generation_functions[session_id]
        
        # Run the search steps
        for step in range(num_steps):
            search_tree = algo.step(search_tree, generate_fns)
            config.current_step += 1
            
            if log_progress and (step + 1) % max(1, num_steps // 5) == 0:
                # Get current best result for progress logging
                try:
                    best_results = tq.top_k(search_tree, algo, k=1)
                    if best_results:
                        best_state, best_score = best_results[0]
                        print(f"Step {config.current_step}: Best score = {best_score:.4f}")
                except:
                    pass  # Continue even if top_k fails
        
        # Update stored state
        state_data["search_tree"] = search_tree
        state_data["config"] = config
        
        # Get final statistics
        try:
            all_pairs = algo.get_state_score_pairs(search_tree)
            total_nodes = len(all_pairs)
            
            # Get top results
            top_results = tq.top_k(search_tree, algo, k=min(5, total_nodes))
            
            result = {
                "status": "success",
                "session_id": session_id,
                "steps_completed": num_steps,
                "total_steps": config.current_step,
                "total_nodes": total_nodes,
                "top_results": [
                    {
                        "state": str(state)[:200] + ("..." if len(str(state)) > 200 else ""),
                        "score": float(score)
                    }
                    for state, score in top_results
                ],
                "generation_functions": list(generate_fns.keys())
            }
            
        except Exception as e:
            result = {
                "status": "partial_success",
                "session_id": session_id,
                "steps_completed": num_steps,
                "total_steps": config.current_step,
                "warning": f"Could not retrieve full statistics: {str(e)}",
                "generation_functions": list(generate_fns.keys())
            }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "session_id": session_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        return json.dumps(error_result, indent=2)


@mcp.tool()
def get_top_results(
    session_id: str,
    k: int = 5
) -> str:
    """
    Get the top-k results from a search session.
    
    Args:
        session_id: The search session ID
        k: Number of top results to retrieve
    
    Returns:
        JSON string with top-k results
    """
    _check_treequest_available()
    
    if session_id not in _search_states:
        return json.dumps({
            "status": "error",
            "error": f"Session '{session_id}' not found. Initialize a search session first."
        })
    
    try:
        state_data = _search_states[session_id]
        algo = state_data["algorithm"]
        search_tree = state_data["search_tree"]
        config = state_data["config"]
        
        # Get all state-score pairs
        all_pairs = algo.get_state_score_pairs(search_tree)
        total_nodes = len(all_pairs)
        
        if total_nodes == 0:
            return json.dumps({
                "status": "success",
                "session_id": session_id,
                "total_nodes": 0,
                "top_results": [],
                "message": "No nodes generated yet. Run search steps first."
            })
        
        # Get top-k results
        k = min(k, total_nodes)  # Don't ask for more than available
        top_results = tq.top_k(search_tree, algo, k=k)
        
        result = {
            "status": "success",
            "session_id": session_id,
            "algorithm_type": config.algorithm_type,
            "total_steps": config.current_step,
            "total_nodes": total_nodes,
            "k": k,
            "top_results": [
                {
                    "rank": i + 1,
                    "state": str(state),
                    "score": float(score),
                    "state_type": type(state).__name__
                }
                for i, (state, score) in enumerate(top_results)
            ]
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "session_id": session_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        return json.dumps(error_result, indent=2)


@mcp.tool()
def list_sessions() -> str:
    """
    List all active search sessions.
    
    Returns:
        JSON string with information about all active sessions
    """
    _check_treequest_available()
    
    sessions = []
    for session_id, state_data in _search_states.items():
        config = state_data["config"]
        algo = state_data["algorithm"]
        search_tree = state_data["search_tree"]
        
        try:
            total_nodes = len(algo.get_state_score_pairs(search_tree))
        except:
            total_nodes = 0
        
        session_info = {
            "session_id": session_id,
            "algorithm_type": config.algorithm_type,
            "algorithm_params": config.algorithm_params,
            "current_step": config.current_step,
            "total_nodes": total_nodes,
            "generation_functions": list(_generation_functions.get(session_id, {}).keys())
        }
        sessions.append(session_info)
    
    result = {
        "status": "success",
        "total_sessions": len(sessions),
        "sessions": sessions
    }
    
    return json.dumps(result, indent=2)


@mcp.tool()
def save_session(
    session_id: str,
    filepath: Optional[str] = None
) -> str:
    """
    Save a search session to disk.
    
    Args:
        session_id: The search session ID to save
        filepath: Optional file path (defaults to session_id.pkl)
    
    Returns:
        JSON string with save operation status
    """
    _check_treequest_available()
    
    if session_id not in _search_states:
        return json.dumps({
            "status": "error",
            "error": f"Session '{session_id}' not found."
        })
    
    try:
        if filepath is None:
            filepath = f"{session_id}.pkl"
        
        save_data = {
            "state_data": _search_states[session_id],
            "generation_functions_info": {
                "action_names": list(_generation_functions.get(session_id, {}).keys()),
                "note": "Generation functions cannot be serialized and must be re-added after loading"
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        result = {
            "status": "success",
            "session_id": session_id,
            "filepath": filepath,
            "message": f"Session saved to {filepath}. Note: Generation functions must be re-added after loading."
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "session_id": session_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        return json.dumps(error_result, indent=2)


@mcp.resource("treequest://docs/{topic}")
def get_documentation(topic: str) -> str:
    """Get documentation for TreeQuest algorithms and usage."""
    
    docs = {
        "overview": """
# TreeQuest Overview

TreeQuest is a flexible answer tree search library featuring AB-MCTS (Adaptive Branching Monte Carlo Tree Search), 
useful for LLM inference-time scaling.

## Key Algorithms:

### AB-MCTS-A (Adaptive Branching with Node Aggregation)
- Uses node aggregation for adaptive branching
- Good for scenarios where you want to balance exploration vs exploitation
- Initialize with: `initialize_search("ABMCTSA", "session_id")`

### AB-MCTS-M (Adaptive Branching with Mixed Models) 
- Leverages PyMC's mixed modeling capabilities
- Requires extra dependencies: install with `treequest[abmcts-m]`
- Initialize with: `initialize_search("ABMCTSM", "session_id")`

### Standard MCTS
- Traditional Monte Carlo Tree Search with UCT scoring
- Good baseline algorithm for tree search problems
- Initialize with: `initialize_search("StandardMCTS", "session_id")`

### Tree of Thoughts BFS
- Breadth-first search variant for tree exploration
- Initialize with: `initialize_search("TreeOfThoughtsBFS", "session_id")`

## Basic Workflow:
1. Initialize a search session with an algorithm
2. Add generation functions for different actions/LLMs
3. Run search steps to explore the tree
4. Get top-k results

See specific topics for detailed information.
""",
        
        "algorithms": """
# TreeQuest Algorithms

## AB-MCTS-A Parameters:
- No specific parameters (uses defaults)

## AB-MCTS-M Parameters:  
- No specific parameters (uses defaults)

## StandardMCTS Parameters:
- `samples_per_action` (int, default=2): Number of samples per action
- `exploration_weight` (float, default=sqrt(2)): UCT exploration weight

## TreeOfThoughtsBFS Parameters:
- Check TreeQuest documentation for specific parameters

## Usage Example:
```python
# Initialize with custom parameters
initialize_search(
    "StandardMCTS", 
    "my_session",
    {"samples_per_action": 3, "exploration_weight": 1.4}
)
```
""",
        
        "generation_functions": """
# Generation Functions

Generation functions are the core of TreeQuest - they define how new states are created from existing ones.

## Function Signature:
```python
def generate(parent_state: Optional[StateType]) -> Tuple[StateType, float]:
    # parent_state is None for root node expansion
    # Return (new_state, score) where score is in [0, 1]
    pass
```

## Adding Functions:

### Simple Functions (for demos):
```python
add_generation_function(
    session_id="my_session",
    action_name="creative_writing", 
    function_type="simple",
    prompt="Write a creative story",
    score_range=(0.0, 1.0)
)
```

### Custom Functions:
```python
custom_code = '''
def generate(parent_state):
    if parent_state is None:
        new_state = "Initial answer"
    else:
        new_state = f"Refined: {parent_state}"
    
    score = 0.8  # Your scoring logic here
    return new_state, score
'''

add_generation_function(
    session_id="my_session",
    action_name="custom_action",
    function_type="custom", 
    custom_code=custom_code
)
```

## Multi-LLM Setup:
Add multiple generation functions with different action names to simulate different LLMs or strategies.
""",

        "examples": """
# TreeQuest Examples

## Basic Search Example:
```python
# 1. Initialize search
initialize_search("StandardMCTS", "demo_session")

# 2. Add generation function
add_generation_function(
    "demo_session",
    "explore", 
    "simple",
    "Explore this problem space",
    (0.0, 1.0)
)

# 3. Run search
run_search_steps("demo_session", 10)

# 4. Get results
get_top_results("demo_session", 3)
```

## Multi-Action Search:
```python
# Initialize
initialize_search("ABMCTSA", "multi_session")

# Add multiple actions
add_generation_function("multi_session", "creative", "simple", "Be creative", (0.3, 1.0))
add_generation_function("multi_session", "analytical", "simple", "Be analytical", (0.2, 0.9))
add_generation_function("multi_session", "practical", "simple", "Be practical", (0.4, 0.8))

# Run search with multiple actions
run_search_steps("multi_session", 20)
get_top_results("multi_session", 5)
```

## LLM Integration Example:
```python
# Custom code for LLM integration
llm_code = '''
import random

def generate(parent_state):
    if parent_state is None:
        # Initial LLM call
        new_state = "LLM generated initial response"
        score = random.uniform(0.6, 0.9)
    else:
        # Refinement LLM call
        new_state = f"LLM refined: {parent_state[:50]}..."
        score = random.uniform(0.7, 1.0)
    
    return new_state, score
'''

add_generation_function("session", "llm_action", "custom", custom_code=llm_code)
```
"""
    }
    
    if topic in docs:
        return docs[topic]
    else:
        available_topics = ", ".join(docs.keys())
        return f"Topic '{topic}' not found. Available topics: {available_topics}"


@mcp.resource("treequest://session/{session_id}/info")  
def get_session_info(session_id: str) -> str:
    """Get detailed information about a specific search session."""
    
    if not TREEQUEST_AVAILABLE:
        return f"TreeQuest is not available. Import error: {IMPORT_ERROR}"
    
    if session_id not in _search_states:
        return f"Session '{session_id}' not found. Available sessions: {list(_search_states.keys())}"
    
    try:
        state_data = _search_states[session_id]
        config = state_data["config"]
        algo = state_data["algorithm"]
        search_tree = state_data["search_tree"]
        
        # Get statistics
        try:
            all_pairs = algo.get_state_score_pairs(search_tree)
            total_nodes = len(all_pairs)
            if total_nodes > 0:
                scores = [score for _, score in all_pairs]
                max_score = max(scores)
                min_score = min(scores)
                avg_score = sum(scores) / len(scores)
            else:
                max_score = min_score = avg_score = 0.0
        except:
            total_nodes = max_score = min_score = avg_score = 0
        
        generation_functions = _generation_functions.get(session_id, {})
        
        info = f"""
# Search Session: {session_id}

## Configuration:
- Algorithm: {config.algorithm_type}
- Algorithm Parameters: {config.algorithm_params}
- Current Step: {config.current_step}
- Total Nodes Generated: {total_nodes}

## Statistics:
- Max Score: {max_score:.4f}
- Min Score: {min_score:.4f}
- Average Score: {avg_score:.4f}

## Generation Functions:
{chr(10).join(f"- {name}" for name in generation_functions.keys()) if generation_functions else "None defined"}

## Status:
{"Ready for search steps" if generation_functions else "Needs generation functions before running search"}
        """
        
        return info.strip()
        
    except Exception as e:
        return f"Error getting session info: {str(e)}"


@mcp.prompt()
def create_llm_generation_function(
    task_description: str,
    llm_model: str = "gpt-4",
    scoring_strategy: str = "quality"
) -> str:
    """
    Create a prompt for implementing an LLM-based generation function for TreeQuest.
    
    Args:
        task_description: Description of the task the LLM should perform
        llm_model: The LLM model to use (e.g., "gpt-4", "claude-3", "gemini-pro")
        scoring_strategy: How to score responses ("quality", "relevance", "creativity", "custom")
    """
    
    return f"""
# LLM Generation Function for TreeQuest

Create a generation function for TreeQuest that uses {llm_model} to solve: **{task_description}**

## Implementation Template:

```python
import openai  # or your preferred LLM client
import json

def generate(parent_state):
    '''
    Generate new states using {llm_model} for: {task_description}
    '''
    
    if parent_state is None:
        # Initial generation from root
        prompt = '''
        Task: {task_description}
        
        Please provide your initial response to this task.
        Return your response in this JSON format:
        {{
            "response": "your response here",
            "confidence": 0.8,
            "reasoning": "brief explanation of your approach"
        }}
        '''
        
    else:
        # Refinement generation
        prompt = f'''
        Task: {task_description}
        
        Previous response: {{parent_state}}
        
        Please refine and improve the previous response.
        Return your improved response in this JSON format:
        {{
            "response": "your improved response here", 
            "confidence": 0.9,
            "reasoning": "brief explanation of improvements made"
        }}
        '''
    
    # Call LLM API
    response = openai.chat.completions.create(
        model="{llm_model}",
        messages=[{{"role": "user", "content": prompt}}],
        temperature=0.7
    )
    
    # Parse response
    try:
        result = json.loads(response.choices[0].message.content)
        new_state = result["response"]
        confidence = result["confidence"]
        
        # Score based on {scoring_strategy}
        score = calculate_score(new_state, confidence, "{scoring_strategy}")
        
    except:
        # Fallback if JSON parsing fails
        new_state = response.choices[0].message.content
        score = 0.5  # Default score
    
    return new_state, score

def calculate_score(response, confidence, strategy):
    '''Calculate score based on scoring strategy'''
    
    if strategy == "quality":
        # Score based on response length, confidence, and basic quality metrics
        length_score = min(len(response) / 500, 1.0)  # Normalize by expected length
        return (confidence * 0.7) + (length_score * 0.3)
        
    elif strategy == "relevance":
        # Score based on relevance to task (implement your logic)
        return confidence * 0.9  # Use confidence as proxy
        
    elif strategy == "creativity":
        # Score based on creativity metrics
        unique_words = len(set(response.lower().split()))
        creativity_score = min(unique_words / 100, 1.0)
        return (confidence * 0.5) + (creativity_score * 0.5)
        
    else:  # custom
        # Implement your custom scoring logic here
        return confidence
```

## Usage Instructions:

1. **Set up your LLM client** (OpenAI, Anthropic, etc.)
2. **Customize the prompts** for your specific task
3. **Implement scoring logic** that matches your evaluation criteria  
4. **Test the function** with sample inputs
5. **Add to TreeQuest** using:

```python
add_generation_function(
    session_id="your_session",
    action_name="{llm_model}_generator",
    function_type="custom",
    custom_code=your_function_code
)
```

## Tips:
- Ensure scores are normalized to [0, 1] range
- Consider using multiple generation functions for different strategies
- Add error handling for API failures
- Log generation costs and track API usage
- Experiment with different temperature settings for exploration vs exploitation
"""


@mcp.prompt()
def setup_multi_llm_search(
    task_description: str,
    llm_models: List[str],
    search_budget: int = 20
) -> str:
    """
    Create a prompt for setting up a multi-LLM tree search with TreeQuest.
    
    Args:
        task_description: Description of the task to solve
        llm_models: List of LLM models to use (e.g., ["gpt-4", "claude-3", "gemini-pro"])
        search_budget: Number of search steps to run
    """
    
    models_str = ", ".join(llm_models)
    
    return f"""
# Multi-LLM Tree Search Setup

Set up a tree search using multiple LLMs ({models_str}) to solve: **{task_description}**

## Complete Setup Code:

```python
import treequest as tq

# 1. Initialize the search session
initialize_search(
    algorithm_type="ABMCTSA",  # Adaptive branching works well for multi-LLM
    session_id="multi_llm_search",
    algorithm_params={{}}  # Use defaults
)

# 2. Add generation functions for each LLM
llm_models = {llm_models}

for model in llm_models:
    # Create model-specific generation function
    model_code = f'''
def generate(parent_state):
    # Import your LLM client here
    
              if parent_state is None:
         prompt = "Task: {task_description}\\\\n\\\\nProvide your initial solution."
     else:
         prompt = f"Task: {task_description}\\\\n\\\\nPrevious attempt: {{{{parent_state}}}}\\\\n\\\\nImprove this solution."
     
     # Call the specific LLM model ({{model}})
     # response = your_llm_client.generate(model="{{model}}", prompt=prompt)
     
     # For demo purposes:
     import random
     new_state = f"{{model}} response to: {{prompt[:50]}}..."
    score = random.uniform(0.5, 1.0)  # Replace with actual scoring
    
    return new_state, score
    '''
    
    add_generation_function(
        session_id="multi_llm_search",
        action_name=f"{{model}}_generator",
        function_type="custom",
        custom_code=model_code
    )

# 3. Run the search
run_search_steps("multi_llm_search", {search_budget})

# 4. Get the best results
get_top_results("multi_llm_search", k=5)
```

## Advanced Multi-LLM Strategy:

### Different Roles for Different Models:
```python
# Specialized generation functions
strategies = {{
    "{llm_models[0] if llm_models else 'gpt-4'}": "creative_exploration",
    "{llm_models[1] if len(llm_models) > 1 else 'claude-3'}": "analytical_refinement", 
    "{llm_models[2] if len(llm_models) > 2 else 'gemini-pro'}": "practical_implementation"
}}

 for model, strategy in strategies.items():
     specialized_code = f'''
 def generate(parent_state):
     strategy = "{{strategy}}"
    
    if strategy == "creative_exploration":
        # Focus on novel approaches
        if parent_state is None:
            prompt = "Think creatively about: {task_description}"
        else:
            prompt = f"Explore creative alternatives to: {{parent_state}}"
            
    elif strategy == "analytical_refinement":
        # Focus on logical improvement
        if parent_state is None:
            prompt = "Analyze this systematically: {task_description}"
        else:
            prompt = f"Analytically improve: {{parent_state}}"
            
    elif strategy == "practical_implementation":
        # Focus on actionable solutions
        if parent_state is None:
            prompt = "Provide practical solution for: {task_description}"
        else:
            prompt = f"Make this more practical: {{parent_state}}"
    
    # Call your LLM here with the specialized prompt
    # ...
    
    return new_state, score
    '''
    
    add_generation_function("multi_llm_search", f"{{model}}_{{strategy}}", "custom", custom_code=specialized_code)
```

## Expected Benefits:
- **Diverse exploration**: Different models explore different solution paths
- **Complementary strengths**: Leverage each model's unique capabilities  
- **Robust solutions**: Cross-validation across multiple AI systems
- **Better coverage**: Avoid single-model biases and limitations

## Monitoring Progress:
```python
# Check progress during search
list_sessions()  # See all active sessions
get_session_info("multi_llm_search")  # Detailed session info
get_top_results("multi_llm_search", k=3)  # Current best results
```

## Tips:
1. **Balance the budget** across models based on their strengths
2. **Customize scoring** to favor diverse high-quality responses
3. **Monitor costs** as multiple LLM calls can be expensive
4. **Experiment with different algorithms** (AB-MCTS-A vs StandardMCTS)
5. **Save sessions** for later analysis and continuation
"""


if __name__ == "__main__":
    mcp.run()