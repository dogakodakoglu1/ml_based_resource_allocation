# Graph Algorithm Resource Allocation ML

A machine learning system designed to forecast the runtime of graph algorithms and optimize the distribution of resources across computing clusters.

## Overview

This project focuses on training machine learning models to estimate execution times for key graph algorithms like PageRank, Connected Components, and Triangle Counting. Additionally, it autonomously identifies the optimal resource allocation required to meet specific deadline constraints.

## Key Features

- **Algorithm Performance Prediction**: Uses an XGBoost model to estimate runtime based on the characteristics of the graph and the configuration of resources.
- **Scheduler Comparison**: Analyzes the ML model's performance against established schedulers (YARN, Spark, Linear Scaling).
- **Deadline-Aware Optimization**: Determines the minimum resources necessary to adhere to time limits.
- **Comprehensive Analysis**: Evaluates memory thresholds, multi-core efficiency, and tests for failure conditions.

## Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost networkx matplotlib seaborn joblib
pip install pyspark  # Optional, for distributed execution
```

### Basic Usage

1. **Run baseline experiments**:
```bash
python enhanced.py
```

2. **Train the ML model**:
```bash
python ml_model_training.py
```

3. **Generate visualizations**:
```bash
python complete_visualizations.py
```

### Project Structure

- `enhanced.py` - Conducts baseline experiments with graph algorithms
- `ml_model_training.py` - Handles ML model training and optimization
- `baseline_schedulers.py` - Implementations for scheduler comparisons
- `complete_visualizations.py` - Produces comprehensive analysis charts
- `memory_threshold_testing.py` - Analyzes resource requirements
## Results

The ML model demonstrates:
- **100% success rate**, outperforming traditional schedulers which are at 75%
- **25% increase in reliability** for workloads constrained by deadlines
- Balanced resource utilization delivering superior results

## Data

To use the system, place graph datasets (in .txt edge list format) in the `data/raw/` directory. The tool automatically evaluates graph properties and selects the most suitable test configurations.

## Output / Artifacts

All generated artifacts are currently stored in the **repository root**:

- **Experiment outputs**
  - `baseline_results.csv`
  - `scheduler_comparison_results.csv`

- **Saved model**
  - `xgboost_model_deadline_aware.joblib`

- **Figures / charts**
  - `baseline_overview.png`
  - `comprehensive_analysis.png`
  - `feature_correlation_analysis.png`
  - `ml_model_performance.png`
  - `model_performance.png`
  - `optimization_results.png`
  - `scheduler_comparison.png`
