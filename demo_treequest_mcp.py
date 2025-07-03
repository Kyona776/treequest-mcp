#!/usr/bin/env python3
"""
TreeQuest MCP Server - Demonstration Script

This script demonstrates the TreeQuest MCP server in action, solving a 
realistic problem using tree search with multiple generation strategies.
"""

import sys
import time
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import treequest as tq


def demo_problem_solving():
    """
    Demonstrate TreeQuest solving a creative problem:
    "Design a sustainable transportation solution for a small city"
    """
    print("🌟 TreeQuest MCP Server Demonstration")
    print("=" * 50)
    print("Problem: Design a sustainable transportation solution for a small city")
    print()
    
    # 1. Initialize search with AB-MCTS-A (good for multiple strategies)
    print("📋 Step 1: Initialize Search")
    algo = tq.ABMCTSA()
    search_tree = algo.init_tree()
    print("✅ AB-MCTS-A algorithm initialized")
    print()
    
    # 2. Define generation functions (simulating different LLM approaches)
    print("🔧 Step 2: Define Generation Strategies")
    
    def generate_environmental_focus(parent_state):
        """Generate environmentally-focused solutions."""
        if parent_state is None:
            solutions = [
                "Electric bus network with solar charging stations",
                "Bike-sharing system with renewable energy maintenance",
                "Walking paths integrated with green corridors"
            ]
        else:
            solutions = [
                f"Enhanced {parent_state} with carbon offset programs",
                f"Expanded {parent_state} with electric vehicle integration",
                f"Optimized {parent_state} with smart grid technology"
            ]
        
        import random
        solution = random.choice(solutions)
        # Environmental solutions get higher scores
        score = random.uniform(0.6, 0.9)
        return solution, score
    
    def generate_economic_focus(parent_state):
        """Generate economically-focused solutions."""
        if parent_state is None:
            solutions = [
                "Public-private partnership for mixed transportation",
                "Ride-sharing platform with local business integration",
                "Affordable micro-transit system"
            ]
        else:
            solutions = [
                f"Cost-optimized {parent_state} with revenue streams",
                f"Scaled {parent_state} with community partnerships",
                f"Efficient {parent_state} with shared infrastructure"
            ]
        
        import random
        solution = random.choice(solutions)
        # Economic solutions get moderate scores
        score = random.uniform(0.4, 0.8)
        return solution, score
    
    def generate_technology_focus(parent_state):
        """Generate technology-focused solutions."""
        if parent_state is None:
            solutions = [
                "Autonomous shuttle network with AI routing",
                "Smart mobility app with real-time optimization",
                "Connected vehicle ecosystem with IoT sensors"
            ]
        else:
            solutions = [
                f"AI-enhanced {parent_state} with predictive analytics",
                f"Automated {parent_state} with machine learning",
                f"Connected {parent_state} with smart city integration"
            ]
        
        import random
        solution = random.choice(solutions)
        # Tech solutions get variable scores
        score = random.uniform(0.3, 0.85)
        return solution, score
    
    def generate_social_focus(parent_state):
        """Generate socially-focused solutions."""
        if parent_state is None:
            solutions = [
                "Community-driven transportation cooperative",
                "Inclusive mobility service for all demographics",
                "Neighborhood-based transportation hubs"
            ]
        else:
            solutions = [
                f"Community-enhanced {parent_state} with local input",
                f"Accessible {parent_state} with universal design",
                f"Equitable {parent_state} with fair pricing"
            ]
        
        import random
        solution = random.choice(solutions)
        # Social solutions get consistent scores
        score = random.uniform(0.5, 0.8)
        return solution, score
    
    actions = {
        'environmental': generate_environmental_focus,
        'economic': generate_economic_focus,
        'technology': generate_technology_focus,
        'social': generate_social_focus
    }
    
    print("✅ Generation strategies defined:")
    print("  - Environmental focus (sustainability)")
    print("  - Economic focus (cost-effectiveness)")
    print("  - Technology focus (innovation)")
    print("  - Social focus (community impact)")
    print()
    
    # 3. Run tree search
    print("🔍 Step 3: Execute Tree Search")
    num_steps = 15
    print(f"Running {num_steps} search steps...")
    
    for step in range(num_steps):
        search_tree = algo.step(search_tree, actions)
        if step % 5 == 4:  # Progress update every 5 steps
            results = tq.top_k(search_tree, algo, k=1)
            if results:
                best_state, best_score = results[0]
                print(f"  Step {step + 1}: Best solution so far: {best_state[:60]}... (Score: {best_score:.3f})")
    
    print("✅ Search completed!")
    print()
    
    # 4. Get and display results
    print("🏆 Step 4: Top Solutions")
    top_results = tq.top_k(search_tree, algo, k=5)
    
    print("Top 5 transportation solutions:")
    print("-" * 40)
    
    for i, (solution, score) in enumerate(top_results, 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"   Solution: {solution}")
        print()
    
    # 5. Analysis
    print("📊 Step 5: Analysis")
    print(f"Total solutions explored: {len(top_results)}")
    if top_results:
        avg_score = sum(score for _, score in top_results) / len(top_results)
        best_score = top_results[0][1]
        print(f"Average score: {avg_score:.3f}")
        print(f"Best score: {best_score:.3f}")
        print(f"Score improvement: {((best_score / avg_score) - 1) * 100:.1f}%")
    
    print()
    print("✨ This demonstrates how TreeQuest MCP server enables:")
    print("  - Multi-strategy exploration (environmental, economic, tech, social)")
    print("  - Adaptive search (AB-MCTS-A algorithm)")
    print("  - Quality ranking and optimization")
    print("  - Scalable solution space exploration")


def demo_mcp_integration():
    """
    Demonstrate how this would work in MCP environment
    """
    print("\n" + "=" * 50)
    print("🔧 MCP Integration Example")
    print("=" * 50)
    
    print("In a real MCP environment, this workflow would be:")
    print()
    
    print("1. 🚀 Initialize search session:")
    print('   initialize_search("ABMCTSA", "transport_problem")')
    print()
    
    print("2. 🤖 Add LLM generation functions:")
    print('   add_generation_function("transport_problem", "environmental", "custom", llm_code)')
    print('   add_generation_function("transport_problem", "economic", "custom", llm_code)')
    print('   add_generation_function("transport_problem", "technology", "custom", llm_code)')
    print('   add_generation_function("transport_problem", "social", "custom", llm_code)')
    print()
    
    print("3. 🔍 Execute search:")
    print('   run_search_steps("transport_problem", 15)')
    print()
    
    print("4. 📊 Get results:")
    print('   get_top_results("transport_problem", 5)')
    print()
    
    print("5. 💾 Save session:")
    print('   save_session("transport_problem", "transport_solutions.pkl")')
    print()
    
    print("🎯 Benefits of TreeQuest MCP approach:")
    print("  ✅ Multiple LLM strategies working together")
    print("  ✅ Intelligent exploration of solution space")
    print("  ✅ Quality-based ranking and optimization")
    print("  ✅ Session persistence and management")
    print("  ✅ Scalable to complex problems")


def main():
    """Run the complete demonstration."""
    print("🌟 Welcome to TreeQuest MCP Server Demo!")
    print()
    print("This demo shows how TreeQuest can solve complex problems")
    print("using tree search with multiple generation strategies.")
    print()
    
    try:
        # Run problem-solving demo
        demo_problem_solving()
        
        # Show MCP integration
        demo_mcp_integration()
        
        print("\n" + "=" * 50)
        print("🎉 Demonstration Complete!")
        print("=" * 50)
        print()
        print("The TreeQuest MCP server is ready for:")
        print("  🔧 Integration with MCP-compatible clients")
        print("  🤖 LLM inference-time scaling")
        print("  🧠 Complex problem-solving workflows")
        print("  🔬 Research and experimentation")
        print()
        print("To start the MCP server: python treequest_mcp_server.py")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())