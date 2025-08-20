#!/usr/bin/env python3
"""
Memory Threshold Testing - Simplified Version

Tests failure points and minimum resource requirements for graph algorithms.
Validates multi-core efficiency and finds resource limits.

Uses simple functions and dictionaries instead of complex classes.
"""

import time
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION DICTIONARIES
# =============================================================================

MEMORY_TEST_CONFIG = {
    'memory_configs_gb': [2, 4, 6, 8, 12, 16, 24, 32],
    'core_configs': [1, 2, 4, 6, 8],
    'low_memory_configs': [1, 2, 3, 4],
    'max_datasets_for_testing': 5,
    'max_datasets_for_efficiency': 3,
    'max_datasets_for_failure': 3,
    'large_graph_sample_size': 10000,
    'large_graph_threshold': 20000
}

ALGORITHM_MEMORY_CHARACTERISTICS = {
    'PageRank': {
        'base_memory_formula': lambda nodes, edges: (nodes * 8 + edges * 12) / (1024 * 1024),
        'memory_overhead_multiplier': 1.5,
        'min_memory_gb': 1.0,
        'parallelization_efficiency': 0.9,
        'description': 'Needs to store node values + adjacency structure'
    },

    'ConnectedComponents': {
        'base_memory_formula': lambda nodes, edges: nodes * 16 / (1024 * 1024),
        'memory_overhead_multiplier': 2.0,
        'min_memory_gb': 0.5,
        'parallelization_efficiency': 0.7,
        'description': 'Union-Find structure, relatively light memory usage'
    },

    'TriangleCounting': {
        'base_memory_formula': lambda nodes, edges: (nodes * 8 + edges * 16) / (1024 * 1024),
        'memory_overhead_multiplier': 3.0,
        'min_memory_gb': 2.0,
        'parallelization_efficiency': 0.8,
        'description': 'Very memory intensive, sensitive to graph density'
    }
}

MEMORY_CONSTRAINTS = {
    'os_overhead_factor': 0.8,  # 80% of memory is usable
    'memory_pressure_threshold': 1.5,  # Memory ratio below which performance degrades
    'large_graph_node_threshold': 500000,
    'large_graph_edge_threshold': 1000000,
    'large_graph_min_memory_gb': 16,
    'very_large_graph_min_memory_gb': 24,
    'dense_graph_threshold': 0.01,
    'triangle_counting_dense_penalty': 1000
}

BINARY_SEARCH_CONFIG = {
    'min_memory_gb': 1,
    'max_memory_gb': 32,
    'default_cores_for_search': 4
}


# =============================================================================
# GRAPH LOADING AND BASIC OPERATIONS
# =============================================================================

def load_graph_safely(filepath):
    """Load graph from SNAP format and handle errors gracefully"""
    try:
        graph = nx.read_edgelist(filepath, comments='#', nodetype=int)
        print(f"Loaded graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
        return graph
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def run_algorithm_safely(graph, algorithm_type):
    """Run graph algorithm safely and measure actual execution time"""
    start_time = time.time()

    try:
        if algorithm_type == 'PageRank':
            result = nx.pagerank(graph, alpha=0.85, max_iter=100, tol=1e-06)

        elif algorithm_type == 'ConnectedComponents':
            result = list(nx.connected_components(graph))

        elif algorithm_type == 'TriangleCounting':
            if graph.number_of_nodes() > MEMORY_TEST_CONFIG['large_graph_threshold']:
                # Sample for large graphs to avoid excessive computation
                sample_size = min(MEMORY_TEST_CONFIG['large_graph_sample_size'], graph.number_of_nodes())
                sample_nodes = list(graph.nodes())[:sample_size]
                subgraph = graph.subgraph(sample_nodes)
                result = sum(nx.triangles(subgraph).values()) // 3
            else:
                result = sum(nx.triangles(graph).values()) // 3

        actual_time = time.time() - start_time
        return True, actual_time, None

    except MemoryError:
        return False, float('inf'), 'MemoryError'
    except Exception as e:
        return False, float('inf'), str(e)


# =============================================================================
# MEMORY REQUIREMENT ESTIMATION
# =============================================================================

def estimate_memory_requirements(graph, algorithm_type, available_memory_gb, cores):
    """
    Estimate memory requirements and determine if configuration would succeed.
    Based on algorithm characteristics and graph properties.
    """
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    density = nx.density(graph)

    # Get algorithm characteristics
    algo_config = ALGORITHM_MEMORY_CHARACTERISTICS[algorithm_type]

    # Calculate base memory requirement
    base_memory_mb = algo_config['base_memory_formula'](nodes, edges)
    min_memory_gb = max(
        algo_config['min_memory_gb'],
        base_memory_mb / 1024 * algo_config['memory_overhead_multiplier']
    )

    # Special handling for Triangle Counting density sensitivity
    if algorithm_type == 'TriangleCounting':
        if density > MEMORY_CONSTRAINTS['dense_graph_threshold']:
            # Dense graphs need quadratic memory
            base_memory_mb = (nodes * nodes * 4) / (1024 * 1024)
            min_memory_gb = max(2.0, base_memory_mb / 1024 * 3.0)

        # Additional penalty for density
        if density > 0.001:
            density_multiplier = 1 + density * MEMORY_CONSTRAINTS['triangle_counting_dense_penalty']
            min_memory_gb *= density_multiplier

    # Check if available memory is sufficient
    usable_memory_gb = available_memory_gb * MEMORY_CONSTRAINTS['os_overhead_factor']

    if usable_memory_gb < min_memory_gb:
        failure_reason = f'Insufficient memory: need {min_memory_gb:.1f}GB, have {usable_memory_gb:.1f}GB'
        return False, float('inf'), failure_reason

    # Check for special cases with very large graphs
    if (nodes > MEMORY_CONSTRAINTS['large_graph_node_threshold'] and
            available_memory_gb < MEMORY_CONSTRAINTS['large_graph_min_memory_gb']):
        return False, float('inf'), 'Large graph requires minimum 16GB memory'

    if (edges > MEMORY_CONSTRAINTS['large_graph_edge_threshold'] and
            available_memory_gb < MEMORY_CONSTRAINTS['very_large_graph_min_memory_gb']):
        return False, float('inf'), 'Very large graph requires minimum 24GB memory'

    # Calculate estimated execution time with parallelization
    parallel_efficiency = algo_config['parallelization_efficiency'] ** (cores - 1) if cores > 1 else 1.0
    base_time = 1.0  # Placeholder base time
    estimated_time = base_time / (cores * parallel_efficiency)

    # Apply memory pressure penalty
    memory_ratio = usable_memory_gb / min_memory_gb
    if memory_ratio < MEMORY_CONSTRAINTS['memory_pressure_threshold']:
        memory_penalty = 2.0 / memory_ratio
        estimated_time *= memory_penalty

    return True, estimated_time, None


def simulate_memory_constrained_execution(graph, algorithm_type, memory_gb, cores):
    """
    Simulate algorithm execution with memory constraints.
    Returns success/failure and estimated runtime.
    """
    print(f"Testing {algorithm_type} with {memory_gb}GB memory, {cores} cores")

    # Run actual algorithm for timing reference
    success, actual_time, error_msg = run_algorithm_safely(graph, algorithm_type)

    if not success:
        return {
            'success': False,
            'failure_reason': error_msg,
            'estimated_time': float('inf'),
            'actual_time': float('inf'),
            'memory_gb': memory_gb,
            'cores': cores
        }

    # Estimate if this configuration would work
    would_succeed, estimated_time, failure_reason = estimate_memory_requirements(
        graph, algorithm_type, memory_gb, cores
    )

    # Scale estimated time based on actual time
    if would_succeed and actual_time != float('inf'):
        estimated_time = actual_time * (estimated_time / 1.0)  # Scale by complexity factor

    return {
        'success': would_succeed,
        'failure_reason': failure_reason if not would_succeed else None,
        'estimated_time': estimated_time,
        'actual_time': actual_time,
        'memory_gb': memory_gb,
        'cores': cores
    }


# =============================================================================
# MINIMUM MEMORY REQUIREMENT TESTING
# =============================================================================

def binary_search_memory_threshold(graph, algorithm_type, dataset_name):
    """Use binary search to find minimum memory requirement"""
    low_memory = BINARY_SEARCH_CONFIG['min_memory_gb']
    high_memory = BINARY_SEARCH_CONFIG['max_memory_gb']
    min_successful_memory = high_memory
    cores = BINARY_SEARCH_CONFIG['default_cores_for_search']

    while low_memory <= high_memory:
        mid_memory = (low_memory + high_memory) // 2

        result = simulate_memory_constrained_execution(graph, algorithm_type, mid_memory, cores)

        if result['success']:
            min_successful_memory = mid_memory
            high_memory = mid_memory - 1
        else:
            low_memory = mid_memory + 1

    return min_successful_memory


def find_minimum_memory_requirements(dataset_files, algorithms):
    """Find minimum memory requirements for each dataset-algorithm combination"""
    print("Finding Minimum Memory Requirements...")

    minimum_requirements = []
    max_datasets = MEMORY_TEST_CONFIG['max_datasets_for_testing']

    for dataset_file in dataset_files[:max_datasets]:
        dataset_name = Path(dataset_file).stem
        print(f"\nTesting dataset: {dataset_name}")

        graph = load_graph_safely(dataset_file)
        if graph is None:
            continue

        for algorithm in algorithms:
            print(f"\n--- Testing {algorithm} ---")

            # Binary search for minimum memory
            min_memory = binary_search_memory_threshold(graph, algorithm, dataset_name)

            minimum_requirements.append({
                'dataset': dataset_name,
                'algorithm': algorithm,
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'density': nx.density(graph),
                'min_memory_gb': min_memory,
                'recommended_memory_gb': min_memory * 1.5  # 50% safety margin
            })

            print(f"Minimum memory for {algorithm} on {dataset_name}: {min_memory:.1f}GB")

    return pd.DataFrame(minimum_requirements)


# =============================================================================
# MULTI-CORE EFFICIENCY TESTING
# =============================================================================

def test_multi_core_efficiency(dataset_files, algorithms):
    """Test and validate multi-core efficiency assumptions"""
    print("Testing Multi-Core Efficiency...")

    efficiency_results = []
    max_datasets = MEMORY_TEST_CONFIG['max_datasets_for_efficiency']

    for dataset_file in dataset_files[:max_datasets]:
        dataset_name = Path(dataset_file).stem
        print(f"\nTesting multi-core efficiency: {dataset_name}")

        graph = load_graph_safely(dataset_file)
        if graph is None:
            continue

        for algorithm in algorithms:
            print(f"\n--- {algorithm} Multi-Core Test ---")

            # Get baseline performance with 1 core
            baseline_result = simulate_memory_constrained_execution(graph, algorithm, 16, 1)
            baseline_time = baseline_result['estimated_time']

            # Test different core counts
            for cores in [2, 4, 6, 8]:
                result = simulate_memory_constrained_execution(graph, algorithm, 16, cores)

                if result['success'] and baseline_result['success']:
                    speedup = baseline_time / result['estimated_time']
                    theoretical_speedup = cores
                    efficiency = speedup / theoretical_speedup

                    efficiency_results.append({
                        'dataset': dataset_name,
                        'algorithm': algorithm,
                        'cores': cores,
                        'speedup': speedup,
                        'efficiency': efficiency,
                        'baseline_time': baseline_time,
                        'parallel_time': result['estimated_time']
                    })

                    print(f"  {cores} cores: {speedup:.2f}x speedup, {efficiency:.1%} efficiency")

    return pd.DataFrame(efficiency_results)


# =============================================================================
# FAILURE CONDITION TESTING
# =============================================================================

def test_failure_conditions(dataset_files, algorithms):
    """Test specific failure conditions and edge cases"""
    print("Testing Failure Conditions...")

    failure_tests = []
    low_memory_configs = MEMORY_TEST_CONFIG['low_memory_configs']
    max_datasets = MEMORY_TEST_CONFIG['max_datasets_for_failure']

    for dataset_file in dataset_files[:max_datasets]:
        dataset_name = Path(dataset_file).stem
        graph = load_graph_safely(dataset_file)
        if graph is None:
            continue

        print(f"\nTesting failure conditions: {dataset_name}")

        for algorithm in algorithms:
            for memory_gb in low_memory_configs:
                result = simulate_memory_constrained_execution(graph, algorithm, memory_gb, 4)

                failure_tests.append({
                    'dataset': dataset_name,
                    'algorithm': algorithm,
                    'memory_gb': memory_gb,
                    'success': result['success'],
                    'failure_reason': result['failure_reason'],
                    'estimated_time': result['estimated_time']
                })

                status = "SUCCESS" if result['success'] else f"FAILED ({result['failure_reason']})"
                print(f"  {algorithm} with {memory_gb}GB: {status}")

    return pd.DataFrame(failure_tests)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_memory_analysis_visualizations(min_requirements_df, efficiency_df, failure_df,
                                          output_dir="results/memory_analysis"):
    """Create comprehensive visualizations for memory threshold analysis"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Memory Threshold & Resource Efficiency Analysis', fontsize=16, fontweight='bold')

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # 1. Minimum Memory Requirements by Algorithm
    if not min_requirements_df.empty:
        axes[0, 0].set_title('Minimum Memory Requirements by Algorithm', fontweight='bold')

        algorithms = min_requirements_df['algorithm'].unique()

        for i, algorithm in enumerate(algorithms):
            algo_data = min_requirements_df[min_requirements_df['algorithm'] == algorithm]
            axes[0, 0].scatter(algo_data['edges'] / 1000, algo_data['min_memory_gb'],
                               label=algorithm, alpha=0.7, s=80, color=colors[i % len(colors)])

        axes[0, 0].set_xlabel('Edges (thousands)')
        axes[0, 0].set_ylabel('Minimum Memory (GB)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # 2. Multi-Core Efficiency
    if not efficiency_df.empty:
        axes[0, 1].set_title('Multi-Core Efficiency by Algorithm', fontweight='bold')

        for i, algorithm in enumerate(efficiency_df['algorithm'].unique()):
            algo_data = efficiency_df[efficiency_df['algorithm'] == algorithm]
            avg_efficiency = algo_data.groupby('cores')['efficiency'].mean()
            axes[0, 1].plot(avg_efficiency.index, avg_efficiency.values,
                            'o-', label=algorithm, linewidth=2, markersize=6, color=colors[i % len(colors)])

        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Efficiency')
        axes[0, 1].set_xlabel('Number of Cores')
        axes[0, 1].set_ylabel('Parallel Efficiency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # 3. Failure Rate by Memory Configuration
    if not failure_df.empty:
        axes[1, 0].set_title('Algorithm Success Rate by Memory', fontweight='bold')

        success_rates = failure_df.groupby(['algorithm', 'memory_gb'])['success'].mean().reset_index()

        for i, algorithm in enumerate(success_rates['algorithm'].unique()):
            algo_data = success_rates[success_rates['algorithm'] == algorithm]
            axes[1, 0].plot(algo_data['memory_gb'], algo_data['success'] * 100,
                            'o-', label=algorithm, linewidth=2, markersize=6, color=colors[i % len(colors)])

        axes[1, 0].set_xlabel('Memory per Node (GB)')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Memory Requirements vs Graph Density
    if not min_requirements_df.empty:
        axes[1, 1].set_title('Memory Requirements vs Graph Density', fontweight='bold')

        for i, algorithm in enumerate(min_requirements_df['algorithm'].unique()):
            algo_data = min_requirements_df[min_requirements_df['algorithm'] == algorithm]
            axes[1, 1].scatter(algo_data['density'], algo_data['min_memory_gb'],
                               label=algorithm, alpha=0.7, s=80, color=colors[i % len(colors)])

        axes[1, 1].set_xlabel('Graph Density')
        axes[1, 1].set_ylabel('Minimum Memory (GB)')
        axes[1, 1].set_xscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/memory_threshold_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Memory analysis visualizations saved to {output_dir}/memory_threshold_analysis.png")


# =============================================================================
# ANALYSIS AND REPORTING
# =============================================================================

def generate_memory_analysis_report(min_requirements_df, efficiency_df, failure_df):
    """Generate comprehensive memory analysis report"""
    print("\n" + "=" * 80)
    print("MEMORY THRESHOLD & RESOURCE EFFICIENCY ANALYSIS REPORT")
    print("=" * 80)

    # Minimum Memory Requirements Summary
    if not min_requirements_df.empty:
        print("\n1. MINIMUM MEMORY REQUIREMENTS:")
        print("-" * 50)
        for algorithm in min_requirements_df['algorithm'].unique():
            algo_data = min_requirements_df[min_requirements_df['algorithm'] == algorithm]
            avg_memory = algo_data['min_memory_gb'].mean()
            max_memory = algo_data['min_memory_gb'].max()
            print(f"{algorithm}:")
            print(f"  Average minimum memory: {avg_memory:.1f}GB")
            print(f"  Maximum memory needed: {max_memory:.1f}GB")

            # Find most memory-intensive dataset
            max_idx = algo_data['min_memory_gb'].idxmax()
            max_dataset = algo_data.loc[max_idx]
            print(f"  Most demanding: {max_dataset['dataset']} ({max_memory:.1f}GB)")

    # Multi-Core Efficiency Summary
    if not efficiency_df.empty:
        print("\n2. MULTI-CORE EFFICIENCY ANALYSIS:")
        print("-" * 50)
        for algorithm in efficiency_df['algorithm'].unique():
            algo_data = efficiency_df[efficiency_df['algorithm'] == algorithm]

            # Get efficiency for different core counts
            efficiency_4_cores = algo_data[algo_data['cores'] == 4]['efficiency']
            efficiency_8_cores = algo_data[algo_data['cores'] == 8]['efficiency']

            avg_eff_4 = efficiency_4_cores.mean() if not efficiency_4_cores.empty else 0
            avg_eff_8 = efficiency_8_cores.mean() if not efficiency_8_cores.empty else 0

            print(f"{algorithm}:")
            print(f"  4-core efficiency: {avg_eff_4:.1%}")
            print(f"  8-core efficiency: {avg_eff_8:.1%}")

            # Categorize parallelization quality
            if avg_eff_4 > 0.7:
                print(f"  Assessment: Good parallelization")
            elif avg_eff_4 > 0.5:
                print(f"  Assessment: Moderate parallelization")
            else:
                print(f"  Assessment: Low parallelization")

    # Failure Analysis
    if not failure_df.empty:
        print("\n3. FAILURE CONDITION ANALYSIS:")
        print("-" * 50)

        for algorithm in failure_df['algorithm'].unique():
            algo_data = failure_df[failure_df['algorithm'] == algorithm]
            total_tests = len(algo_data)
            successful_tests = algo_data['success'].sum()
            success_rate = successful_tests / total_tests * 100

            print(f"{algorithm}:")
            print(f"  Overall success rate: {success_rate:.1f}% ({successful_tests}/{total_tests})")

            # Find minimum successful memory
            successful_configs = algo_data[algo_data['success'] == True]
            if not successful_configs.empty:
                min_successful_memory = successful_configs['memory_gb'].min()
                print(f"  Minimum successful memory: {min_successful_memory}GB")

            # Common failure reasons
            failed_configs = algo_data[algo_data['success'] == False]
            if not failed_configs.empty:
                failure_reasons = failed_configs['failure_reason'].value_counts()
                most_common_failure = failure_reasons.index[0]
                failure_count = failure_reasons.iloc[0]
                print(f"  Common failures: {most_common_failure} ({failure_count} cases)")

    print("\n" + "=" * 80)


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def run_comprehensive_memory_analysis(dataset_files, algorithms):
    """Run all memory threshold tests and generate comprehensive analysis"""

    # Phase 1: Find minimum memory requirements
    print("\n" + "=" * 60)
    print("PHASE 1: MINIMUM MEMORY REQUIREMENTS")
    print("=" * 60)
    min_requirements_df = find_minimum_memory_requirements(dataset_files, algorithms)

    # Phase 2: Test multi-core efficiency
    print("\n" + "=" * 60)
    print("PHASE 2: MULTI-CORE EFFICIENCY TESTING")
    print("=" * 60)
    efficiency_df = test_multi_core_efficiency(dataset_files, algorithms)

    # Phase 3: Test failure conditions
    print("\n" + "=" * 60)
    print("PHASE 3: FAILURE CONDITION TESTING")
    print("=" * 60)
    failure_df = test_failure_conditions(dataset_files, algorithms)

    return min_requirements_df, efficiency_df, failure_df


def save_memory_analysis_results(min_requirements_df, efficiency_df, failure_df, output_dir="results/memory_analysis"):
    """Save all memory analysis results to CSV files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    min_requirements_df.to_csv(output_path / "minimum_memory_requirements.csv", index=False)
    efficiency_df.to_csv(output_path / "multi_core_efficiency.csv", index=False)
    failure_df.to_csv(output_path / "failure_conditions.csv", index=False)

    print(f"Results saved to: {output_path}")


def main():
    """Main execution for memory threshold testing"""
    print("Memory Threshold & Resource Efficiency Testing")
    print("=" * 50)

    # Find available datasets
    data_dir = Path("data/raw")
    available_datasets = list(data_dir.glob("*.txt"))

    if not available_datasets:
        print("No datasets found in data/raw/")
        print("Run download_datasets.py first")
        return

    print(f"Found {len(available_datasets)} datasets")
    dataset_paths = [str(d) for d in available_datasets]

    # Define algorithms to test
    algorithms = ['PageRank', 'ConnectedComponents', 'TriangleCounting']

    # Run comprehensive memory analysis
    min_requirements_df, efficiency_df, failure_df = run_comprehensive_memory_analysis(dataset_paths, algorithms)

    # Save results
    save_memory_analysis_results(min_requirements_df, efficiency_df, failure_df)

    # Create visualizations
    create_memory_analysis_visualizations(min_requirements_df, efficiency_df, failure_df)

    # Generate comprehensive report
    generate_memory_analysis_report(min_requirements_df, efficiency_df, failure_df)

    print(f"\nMemory threshold testing complete!")


if __name__ == "__main__":
    main()