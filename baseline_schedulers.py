#!/usr/bin/env python3
"""
Baseline Scheduler Implementations - Simplified Version

Simulates different scheduling algorithms (YARN, Spark, heuristics) for
comparison with ML model predictions. Uses simple functions and dictionaries
instead of complex classes.

Used by ml_model_training.py for scheduler comparison.
"""

import numpy as np
import pandas as pd
import math

# =============================================================================
# CONFIGURATION DICTIONARIES
# =============================================================================

CLUSTER_CONFIG = {
    'max_nodes': 8,
    'min_nodes': 1,
    'cores_per_node': 4,
    'memory_per_node_gb': 8
}

ALGORITHM_COMPLEXITY = {
    'PageRank': 1.0,  # Baseline complexity
    'ConnectedComponents': 0.8,  # Simpler algorithm
    'TriangleCounting': 5.4  # Much more complex (empirically measured)
}

SCHEDULER_CONFIGS = {
    'yarn': {
        'name': 'YARN_Fair',
        'approach': 'data_driven_conservative',
        'min_allocation': 2,  # YARN's conservative minimum
        'complexity_boost': {'high': 2, 'medium': 1, 'low': 0}
    },

    'spark': {
        'name': 'Spark_Default',
        'approach': 'partition_based_adaptive',
        'min_allocation': 1,  # Spark can start with 1 node
        'partition_edges': 100000,  # Edges per partition
        'density_threshold': 0.001
    },

    'linear': {
        'name': 'Linear_Scaling',
        'approach': 'size_proportional',
        'edges_per_node': 50000,  # Linear scaling threshold
        'complexity_multipliers': {'high': 1.5, 'medium': 1.0, 'low': 0.8}
    },

    'fixed': {
        'name': 'Fixed_4_Nodes',
        'approach': 'conservative_constant',
        'allocation': 4  # Always allocate 4 nodes
    }
}

PARALLELIZATION_EFFICIENCY = {
    'PageRank': 0.9,  # Good parallelization
    'ConnectedComponents': 0.7,  # Poor parallelization (sequential nature)
    'TriangleCounting': 0.8  # Moderate parallelization
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_algorithm_complexity_category(algorithm_type):
    """Categorize algorithm by computational complexity"""
    complexity = ALGORITHM_COMPLEXITY.get(algorithm_type, 1.0)

    if complexity > 3.0:
        return 'high'
    elif complexity > 1.0:
        return 'medium'
    else:
        return 'low'


def calculate_data_size_factor(edges):
    """Calculate a scaling factor based on graph size"""
    return math.log10(max(edges, 1000)) / math.log10(1000)


def ensure_allocation_bounds(allocation):
    """Ensure allocation is within cluster limits"""
    return max(CLUSTER_CONFIG['min_nodes'], min(CLUSTER_CONFIG['max_nodes'], allocation))


def create_allocation_result(scheduler_name, num_nodes, reasoning):
    """Create standardized allocation result dictionary"""
    return {
        'scheduler': scheduler_name,
        'num_nodes': num_nodes,
        'total_cores': num_nodes * CLUSTER_CONFIG['cores_per_node'],
        'total_memory_gb': num_nodes * CLUSTER_CONFIG['memory_per_node_gb'],
        'reasoning': reasoning,
        'cost': num_nodes
    }


# =============================================================================
# SCHEDULER IMPLEMENTATIONS
# =============================================================================

def yarn_fair_scheduler(graph_features, algorithm_type):
    """
    Simulate YARN Fair Scheduler allocation logic.
    YARN uses capacity scheduling and fair sharing principles.
    Conservative approach that allocates more resources for safety.
    """
    nodes = graph_features['nodes']
    edges = graph_features['edges']
    density = graph_features['density']

    config = SCHEDULER_CONFIGS['yarn']

    # YARN's resource estimation based on data size and complexity
    data_size_factor = calculate_data_size_factor(edges)
    base_allocation = min(CLUSTER_CONFIG['max_nodes'], max(2, int(data_size_factor)))

    # Algorithm-specific adjustments (YARN would have job-type profiles)
    complexity_category = get_algorithm_complexity_category(algorithm_type)
    complexity_boost = config['complexity_boost'][complexity_category]

    yarn_allocation = min(CLUSTER_CONFIG['max_nodes'], base_allocation + complexity_boost)

    # Apply YARN's minimum allocation policy
    yarn_allocation = max(config['min_allocation'], yarn_allocation)

    reasoning = f'Data-driven allocation based on {edges:,} edges, algorithm complexity {ALGORITHM_COMPLEXITY[algorithm_type]:.1f}x'

    return create_allocation_result(config['name'], yarn_allocation, reasoning)


def spark_default_scheduler(graph_features, algorithm_type):
    """
    Simulate Spark's default dynamic allocation behavior.
    Based on Spark's adaptive query execution and dynamic allocation.
    Partition-based approach with density awareness.
    """
    nodes = graph_features['nodes']
    edges = graph_features['edges']
    density = graph_features['density']

    config = SCHEDULER_CONFIGS['spark']

    # Spark's partition-based allocation
    estimated_partitions = max(1, edges // config['partition_edges'])
    initial_allocation = min(CLUSTER_CONFIG['max_nodes'], max(1, int(math.sqrt(estimated_partitions))))

    # Spark's algorithm-aware optimization (adaptive query execution)
    if algorithm_type == 'TriangleCounting':
        # Spark recognizes memory-intensive operations
        spark_allocation = min(CLUSTER_CONFIG['max_nodes'], initial_allocation + 2)
    elif algorithm_type == 'PageRank':
        # Iterative algorithms benefit from moderate parallelism
        spark_allocation = min(CLUSTER_CONFIG['max_nodes'], initial_allocation + 1)
    else:  # ConnectedComponents
        # Simple traversal algorithms
        spark_allocation = initial_allocation

    # Apply Spark's minimum allocation
    spark_allocation = max(config['min_allocation'], spark_allocation)

    # Spark's memory-aware adjustment for dense graphs
    if density > config['density_threshold']:
        spark_allocation = min(CLUSTER_CONFIG['max_nodes'], spark_allocation + 1)

    reasoning = f'Partition-based allocation: {estimated_partitions} partitions, density-aware'

    return create_allocation_result(config['name'], spark_allocation, reasoning)


def linear_scaling_scheduler(graph_features, algorithm_type):
    """
    Simple linear scaling heuristic based on graph size.
    Common approach: scale resources linearly with problem size.
    """
    nodes = graph_features['nodes']
    edges = graph_features['edges']

    config = SCHEDULER_CONFIGS['linear']

    # Linear scaling based on graph size
    size_based_allocation = max(1, edges // config['edges_per_node'])

    # Algorithm-specific scaling
    complexity_category = get_algorithm_complexity_category(algorithm_type)
    multiplier = config['complexity_multipliers'][complexity_category]

    if complexity_category == 'high':  # TriangleCounting
        linear_allocation = int(size_based_allocation * multiplier)
    elif complexity_category == 'medium':  # PageRank
        linear_allocation = size_based_allocation
    else:  # ConnectedComponents
        linear_allocation = max(1, int(size_based_allocation * multiplier))

    # Ensure within bounds
    linear_allocation = ensure_allocation_bounds(linear_allocation)

    reasoning = f'Linear scaling: 1 node per {config["edges_per_node"]:,} edges, {ALGORITHM_COMPLEXITY[algorithm_type]:.1f}x complexity adjustment'

    return create_allocation_result(config['name'], linear_allocation, reasoning)


def fixed_allocation_scheduler(graph_features, algorithm_type):
    """
    Conservative fixed allocation approach.
    Always allocate the same amount of resources regardless of workload.
    """
    config = SCHEDULER_CONFIGS['fixed']
    fixed_allocation = config['allocation']

    reasoning = 'Conservative fixed allocation for predictable performance'

    return create_allocation_result(config['name'], fixed_allocation, reasoning)


def optimal_oracle_scheduler(graph_features, algorithm_type):
    """
    Theoretical optimal allocation based on empirical data.
    This represents the "oracle" scheduler that knows the optimal allocation.
    """
    nodes = graph_features['nodes']
    edges = graph_features['edges']

    # Based on baseline experiment results
    if algorithm_type == 'TriangleCounting':
        # TriangleCounting often needs more resources for large/dense graphs
        if edges > 150000:  # Large graphs
            optimal_allocation = 4
        elif edges > 50000:  # Medium graphs
            optimal_allocation = 2
        else:  # Small graphs
            optimal_allocation = 1
    else:  # PageRank and ConnectedComponents
        # These algorithms are generally efficient with minimal resources
        optimal_allocation = 1

    reasoning = 'Oracle allocation based on empirical optimal results'

    return create_allocation_result('Optimal_Oracle', optimal_allocation, reasoning)


# =============================================================================
# SCHEDULER COMPARISON AND EVALUATION
# =============================================================================

def compare_all_schedulers(graph_features, algorithm_type):
    """
    Compare all baseline schedulers for given graph and algorithm.
    Returns comprehensive comparison of different scheduling approaches.
    """
    schedulers = {
        'yarn': yarn_fair_scheduler,
        'spark': spark_default_scheduler,
        'linear': linear_scaling_scheduler,
        'fixed': fixed_allocation_scheduler,
        'optimal': optimal_oracle_scheduler
    }

    results = {}
    for name, scheduler_func in schedulers.items():
        results[name] = scheduler_func(graph_features, algorithm_type)

    return results


def estimate_runtime_for_allocation(allocation_result, graph_features, algorithm_type):
    """
    Estimate runtime for a given allocation using simplified prediction model.
    Used to evaluate scheduler effectiveness.
    """
    num_nodes = allocation_result['num_nodes']
    edges = graph_features['edges']
    graph_nodes = graph_features['nodes']
    density = graph_features['density']

    # Get algorithm characteristics
    complexity_factor = ALGORITHM_COMPLEXITY.get(algorithm_type, 1.0)
    efficiency = PARALLELIZATION_EFFICIENCY.get(algorithm_type, 0.8)

    # Base computation time (edges-based complexity)
    base_time = (edges / 10000) * np.log(graph_nodes) * 0.1 * complexity_factor

    # Calculate parallelization efficiency (diminishing returns)
    scale_factor = 1.0 / (num_nodes * (efficiency ** (num_nodes - 1))) if num_nodes > 1 else 1.0

    # Memory factor (more memory per node = better performance)
    memory_per_node = allocation_result['total_memory_gb'] / num_nodes
    memory_factor = max(0.5, 8.0 / memory_per_node)

    # Algorithm-specific adjustments
    if algorithm_type == 'TriangleCounting':
        # Dense graphs are much harder for triangle counting
        density_factor = 1.0 + (density * 10)
        estimated_time = base_time * scale_factor * memory_factor * density_factor
    else:
        estimated_time = base_time * scale_factor * memory_factor

    # Add some realistic randomness and ensure minimum time
    estimated_time += np.random.normal(0, estimated_time * 0.05)
    estimated_time = max(estimated_time, 0.8)

    return estimated_time


def evaluate_scheduler_performance(graph_features, algorithm_type, target_deadline=60.0):
    """
    Evaluate all schedulers and rank them by cost-effectiveness.
    Returns analysis of which scheduler performs best for given workload.
    """
    scheduler_results = compare_all_schedulers(graph_features, algorithm_type)

    evaluation = {}

    for scheduler_name, allocation in scheduler_results.items():
        estimated_runtime = estimate_runtime_for_allocation(allocation, graph_features, algorithm_type)

        meets_deadline = estimated_runtime <= target_deadline
        cost = allocation['cost']

        # Calculate efficiency score (lower is better)
        if meets_deadline:
            efficiency_score = cost  # Just the cost if deadline is met
        else:
            efficiency_score = cost * 2 + (estimated_runtime - target_deadline)  # Penalty for missing deadline

        evaluation[scheduler_name] = {
            'allocation': allocation,
            'estimated_runtime': estimated_runtime,
            'meets_deadline': meets_deadline,
            'cost': cost,
            'efficiency_score': efficiency_score
        }

    # Sort by efficiency score (lower is better)
    sorted_schedulers = sorted(evaluation.items(), key=lambda x: x[1]['efficiency_score'])

    return evaluation, sorted_schedulers


# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

def test_all_baseline_schedulers():
    """Test function to verify all scheduler implementations work correctly"""
    print("Testing Baseline Schedulers...")

    # Define test graphs with different characteristics
    test_graphs = [
        {
            'name': 'Small Graph',
            'nodes': 5000,
            'edges': 25000,
            'density': 0.002,
            'avg_clustering': 0.05
        },
        {
            'name': 'Medium Graph',
            'nodes': 50000,
            'edges': 200000,
            'density': 0.0001,
            'avg_clustering': 0.1
        },
        {
            'name': 'Large Graph',
            'nodes': 200000,
            'edges': 500000,
            'density': 0.00001,
            'avg_clustering': 0.2
        }
    ]

    algorithms = ['PageRank', 'ConnectedComponents', 'TriangleCounting']

    for graph in test_graphs:
        print(f"\n{graph['name']}:")
        print(f"  {graph['nodes']:,} nodes, {graph['edges']:,} edges")

        for algorithm in algorithms:
            print(f"\n  {algorithm}:")

            # Compare all schedulers
            scheduler_results = compare_all_schedulers(graph, algorithm)

            for scheduler_name, result in scheduler_results.items():
                estimated_runtime = estimate_runtime_for_allocation(result, graph, algorithm)
                print(f"    {result['scheduler']}: {result['num_nodes']} nodes "
                      f"({estimated_runtime:.1f}s estimated)")

    print("\nBaseline Schedulers Test Complete!")


def demonstrate_scheduler_comparison():
    """Demonstrate how to compare schedulers for a specific workload"""
    print("\nScheduler Comparison Demonstration:")
    print("=" * 50)

    # Example workload
    example_graph = {
        'nodes': 75000,
        'edges': 300000,
        'density': 0.0005,
        'avg_clustering': 0.15
    }

    algorithm = 'TriangleCounting'

    print(f"Graph: {example_graph['nodes']:,} nodes, {example_graph['edges']:,} edges")
    print(f"Algorithm: {algorithm}")
    print(f"Target deadline: 60 seconds")

    # Evaluate all schedulers
    evaluation, ranked_schedulers = evaluate_scheduler_performance(example_graph, algorithm)

    print(f"\nScheduler Rankings (by cost-effectiveness):")
    print("-" * 40)

    for i, (scheduler_name, results) in enumerate(ranked_schedulers, 1):
        allocation = results['allocation']
        runtime = results['estimated_runtime']
        meets_deadline = results['meets_deadline']
        cost = results['cost']

        status = "PASS" if meets_deadline else "FAIL"
        print(f"{i}. {allocation['scheduler']}: {allocation['num_nodes']} nodes, "
              f"{runtime:.1f}s, Cost: {cost}, [{status}]")

    best_scheduler = ranked_schedulers[0]
    print(f"\nBest scheduler: {best_scheduler[0]} ({best_scheduler[1]['allocation']['scheduler']})")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to test and demonstrate scheduler functionality"""
    print("Baseline Scheduler Implementations")
    print("=" * 40)

    # Run comprehensive tests
    test_all_baseline_schedulers()

    # Demonstrate scheduler comparison
    demonstrate_scheduler_comparison()

    print("\nAll scheduler tests completed successfully!")


if __name__ == "__main__":
    main()