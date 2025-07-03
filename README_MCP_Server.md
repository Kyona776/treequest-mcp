# TreeQuest MCP Server

A Model Context Protocol (MCP) server that exposes TreeQuest's tree search algorithms for LLM inference-time scaling. Built using fastMCP as the MCP SDK.

## About TreeQuest

TreeQuest is a flexible answer tree search library featuring **AB-MCTS** (Adaptive Branching Monte Carlo Tree Search), useful for LLM inference-time scaling. It provides several powerful algorithms:

- **AB-MCTS-A**: Adaptive Branching with Node Aggregation
- **AB-MCTS-M**: Adaptive Branching with Mixed Models (requires PyMC)
- **StandardMCTS**: Traditional Monte Carlo Tree Search with UCT scoring
- **TreeOfThoughtsBFS**: Breadth-first search variant for tree exploration

## Features

This MCP server provides:

- **Tools** for initializing and running tree search algorithms
- **Resources** for accessing documentation and session information  
- **Prompts** for setting up LLM-based generation functions
- Support for multiple generation functions (multi-LLM setups)
- Session management and checkpointing
- Progress tracking and result ranking

## Installation

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)

### Setup

1. **Clone the repository** (if you have the TreeQuest source):
   ```bash
   git clone <treequest-repo-url>
   cd treequest
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # Core dependencies
   pip install fastmcp scipy
   
   # For AB-MCTS-M algorithm (optional but recommended):
   pip install pandas pymc numpyro jax
   ```

4. **Test the installation**:
   ```bash
   python test_treequest_mcp.py
   ```

## Usage

### Starting the MCP Server

```bash
source venv/bin/activate
python treequest_mcp_server.py
```

The server will start and wait for MCP client connections.

### Available Tools

#### 1. `initialize_search`
Initialize a new tree search session.

**Parameters:**
- `algorithm_type`: Algorithm to use (`ABMCTSA`, `ABMCTSM`, `StandardMCTS`, `TreeOfThoughtsBFS`)
- `session_id`: Unique identifier for the session (default: "default")
- `algorithm_params`: Optional algorithm parameters (e.g., `exploration_weight`, `samples_per_action`)

**Example:**
```python
initialize_search("StandardMCTS", "my_session", {"exploration_weight": 1.4})
```

#### 2. `add_generation_function`
Add a generation function to a search session.

**Parameters:**
- `session_id`: The search session ID
- `action_name`: Name for this action type
- `function_type`: `"simple"` (for demos) or `"custom"` (user-defined code)
- `prompt`: Prompt text for simple functions
- `score_range`: Score range for simple functions (min, max)
- `custom_code`: Python code defining a `generate` function

**Example:**
```python
# Simple function
add_generation_function("my_session", "creative", "simple", "Be creative", (0.3, 1.0))

# Custom function
custom_code = '''
def generate(parent_state):
    if parent_state is None:
        new_state = "Initial answer"
    else:
        new_state = f"Refined: {parent_state}"
    score = 0.8  # Your scoring logic
    return new_state, score
'''
add_generation_function("my_session", "custom_action", "custom", custom_code=custom_code)
```

#### 3. `run_search_steps`
Run multiple steps of the tree search algorithm.

**Parameters:**
- `session_id`: The search session ID
- `num_steps`: Number of search steps to execute
- `log_progress`: Whether to log progress during search

#### 4. `get_top_results`
Get the top-k results from a search session.

**Parameters:**
- `session_id`: The search session ID
- `k`: Number of top results to retrieve (default: 5)

#### 5. `list_sessions`
List all active search sessions.

#### 6. `save_session`
Save a search session to disk.

**Parameters:**
- `session_id`: The search session ID to save
- `filepath`: Optional file path (defaults to `session_id.pkl`)

### Available Resources

#### 1. `treequest://docs/{topic}`
Get documentation for TreeQuest algorithms and usage.

**Available topics:**
- `overview`: General overview of TreeQuest
- `algorithms`: Detailed algorithm documentation
- `generation_functions`: How to create generation functions
- `examples`: Usage examples

#### 2. `treequest://session/{session_id}/info`
Get detailed information about a specific search session.

### Available Prompts

#### 1. `create_llm_generation_function`
Create a prompt for implementing an LLM-based generation function.

**Parameters:**
- `task_description`: Description of the task the LLM should perform
- `llm_model`: The LLM model to use (e.g., "gpt-4", "claude-3")
- `scoring_strategy`: How to score responses ("quality", "relevance", "creativity")

#### 2. `setup_multi_llm_search`
Create a prompt for setting up a multi-LLM tree search.

**Parameters:**
- `task_description`: Description of the task to solve
- `llm_models`: List of LLM models to use
- `search_budget`: Number of search steps to run

## Example Workflows

### Basic Search Example

1. **Initialize a search session**:
   ```python
   initialize_search("StandardMCTS", "demo_session")
   ```

2. **Add generation functions**:
   ```python
   add_generation_function("demo_session", "explore", "simple", "Explore this problem", (0.0, 1.0))
   ```

3. **Run the search**:
   ```python
   run_search_steps("demo_session", 10)
   ```

4. **Get results**:
   ```python
   get_top_results("demo_session", 3)
   ```

### Multi-LLM Search Example

1. **Initialize with AB-MCTS-A** (good for multi-LLM):
   ```python
   initialize_search("ABMCTSA", "multi_llm_session")
   ```

2. **Add multiple generation functions**:
   ```python
   add_generation_function("multi_llm_session", "gpt4_creative", "simple", "Be creative with GPT-4", (0.4, 1.0))
   add_generation_function("multi_llm_session", "claude_analytical", "simple", "Be analytical with Claude", (0.3, 0.9))
   add_generation_function("multi_llm_session", "gemini_practical", "simple", "Be practical with Gemini", (0.5, 0.8))
   ```

3. **Run search with multiple actions**:
   ```python
   run_search_steps("multi_llm_session", 20)
   ```

4. **Monitor progress**:
   ```python
   get_session_info("multi_llm_session")  # Check session details
   get_top_results("multi_llm_session", 5)  # Get best results
   ```

### LLM Integration

For real LLM integration, use the `create_llm_generation_function` prompt to get a template, then implement with your preferred LLM client:

```python
# Use the prompt to get implementation template
create_llm_generation_function("Solve math problems", "gpt-4", "quality")

# Then implement the custom code with your LLM client
custom_llm_code = '''
import openai

def generate(parent_state):
    if parent_state is None:
        prompt = "Solve this math problem: ..."
    else:
        prompt = f"Improve this solution: {parent_state}"
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    new_state = response.choices[0].message.content
    score = calculate_quality_score(new_state)  # Your scoring logic
    
    return new_state, score
'''

add_generation_function("session", "gpt4_math", "custom", custom_code=custom_llm_code)
```

## Algorithm Parameters

### StandardMCTS
- `samples_per_action` (int, default=2): Number of samples per action
- `exploration_weight` (float, default=√2): UCT exploration weight

### AB-MCTS-A and AB-MCTS-M
- Use default parameters (customization available in TreeQuest library)

## Tips for LLM Integration

1. **Score Normalization**: Ensure scores are in [0, 1] range
2. **Multiple Strategies**: Use different generation functions for different approaches
3. **Error Handling**: Add robust error handling for LLM API failures
4. **Cost Monitoring**: Track API usage costs
5. **Temperature Settings**: Experiment with different temperature values for exploration vs exploitation

## Troubleshooting

### Import Errors
- Ensure virtual environment is activated
- Install all dependencies: `pip install fastmcp scipy pandas pymc numpyro jax`

### ABMCTSM Not Available
- Install optional dependencies: `pip install pandas pymc numpyro jax`
- This algorithm requires PyMC and JAX

### Session Not Found
- Check session ID spelling
- Use `list_sessions()` to see active sessions
- Initialize session before adding generation functions

## API Reference

For detailed API documentation, use the resource endpoints:
- `treequest://docs/overview` - General overview
- `treequest://docs/algorithms` - Algorithm details
- `treequest://docs/generation_functions` - Function creation guide
- `treequest://docs/examples` - Usage examples

## License

This MCP server is built on top of TreeQuest and fastMCP. Please refer to their respective licenses.

## Contributing

This is a demonstration MCP server. For TreeQuest improvements, contribute to the main TreeQuest repository. For fastMCP improvements, contribute to the fastMCP project.