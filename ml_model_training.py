#!/usr/bin/env python3
"""
Machine Learning Model Training - Simplified Version

Trains ML models to predict graph algorithm runtime and optimize resource allocation.
Supports multiple deadline constraints and compares with baseline schedulers.

Uses simple functions and dictionaries instead of complex classes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

# ML imports
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# =============================================================================
# CONFIGURATION DICTIONARIES
# =============================================================================

MODEL_CONFIG = {
    'target_column': 'simulated_time',
    'test_size': 0.25,
    'random_state': 42,
    'max_nodes': 10,
    'cores_per_node': 4,
    'memory_per_node_gb': 8
}

FEATURE_COLUMNS = [
    'nodes',  # Graph size (number of nodes)
    'edges',  # Graph size (number of edges)
    'density',  # Graph structure
    'avg_clustering',  # Graph connectivity pattern
    'num_nodes',  # Resource configuration (nodes allocated)
    'total_cores',  # Total computational resources
    'total_memory_gb',  # Total memory resources
    'algorithm_type'  # Algorithm type (PageRank, ConnectedComponents, TriangleCounting)
]

MODEL_ALGORITHMS = {
    'XGBoost': {
        'class': xgb.XGBRegressor,
        'params': {
            'n_estimators': 150,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbosity': 0
        }
    },
    'RandomForest': {
        'class': RandomForestRegressor,
        'params': {
            'n_estimators': 150,
            'max_depth': 8,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
    }
}

VALIDATION_THRESHOLDS = {
    'error_percentage_threshold': 20.0,
    'minimum_realistic_time': 0.1,
    'default_deadline': 60.0
}

DEFAULT_PATHS = {
    'baseline_data': "results/enhanced_baseline/baseline_results.csv",
    'model_save': "results/models/xgboost_model_deadline_aware.joblib",
    'visualizations': "results/models",
    'scheduler_comparison': "results/scheduler_comparison"
}

# =============================================================================
# GLOBAL VARIABLES FOR MODEL STATE
# =============================================================================

model_state = {
    'trained_model': None,
    'scaler': StandardScaler(),
    'label_encoder': LabelEncoder(),
    'feature_names': None,
    'target_name': MODEL_CONFIG['target_column']
}


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_baseline_experiment_data(filepath=None):
    """Load baseline experiment data and display summary statistics"""
    if filepath is None:
        filepath = DEFAULT_PATHS['baseline_data']

    try:
        df = pd.read_csv(filepath)
        print(f"Loaded baseline data: {len(df)} experiments")
        print(f"Datasets: {df['graph_name'].nunique()}")
        print(f"Algorithms: {df['algorithm_type'].nunique()}")

        # Check for deadline constraints
        if 'deadline_constraint' in df.columns:
            deadlines = sorted(df['deadline_constraint'].unique())
            print(f"Deadline constraints tested: {deadlines} seconds")
            experiments_per_combo = len(df) // (
                        df['graph_name'].nunique() * df['algorithm_type'].nunique() * len(deadlines))
            print(f"Configurations per dataset-algorithm-deadline: {experiments_per_combo}")

            # Show deadline-specific success rates
            print(f"\nSuccess rates by deadline:")
            for deadline in deadlines:
                deadline_data = df[df['deadline_constraint'] == deadline]
                success_rate = deadline_data['meets_deadline'].mean() * 100
                success_count = deadline_data['meets_deadline'].sum()
                total_count = len(deadline_data)
                print(f"  {deadline}s: {success_rate:.1f}% ({success_count}/{total_count})")
        else:
            print("Warning: No deadline_constraint column found - using legacy dataset")

        return df
    except Exception as e:
        print(f"Error loading baseline data: {e}")
        return None


def prepare_ml_features(df):
    """Prepare features for ML model training including deadline constraints"""
    print("\nPreparing features for ML model...")

    # Determine which features are available
    available_features = FEATURE_COLUMNS.copy()

    # Add deadline constraint if available
    if 'deadline_constraint' in df.columns:
        available_features.append('deadline_constraint')
        print("Including deadline_constraint as feature")

    # Check for missing features
    missing_features = [f for f in available_features if f not in df.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
        available_features = [f for f in available_features if f in df.columns]

    print(f"Using features: {available_features}")

    # Extract features and target
    X = df[available_features].copy()
    y = df[MODEL_CONFIG['target_column']].copy()

    # Handle categorical algorithm_type feature
    if 'algorithm_type' in X.columns:
        print(f"Algorithm types found: {X['algorithm_type'].unique()}")
        # Encode algorithm_type as numerical
        X['algorithm_type_encoded'] = model_state['label_encoder'].fit_transform(X['algorithm_type'])
        # Remove original categorical column
        X = X.drop('algorithm_type', axis=1)

        # Update feature list
        updated_features = []
        for col in available_features:
            if col == 'algorithm_type':
                updated_features.append('algorithm_type_encoded')
            else:
                updated_features.append(col)
        available_features = updated_features

    # Handle missing values
    X = X.fillna(X.mean())

    # Store feature names for later use
    model_state['feature_names'] = list(X.columns)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Final feature order: {model_state['feature_names']}")
    print(f"Target range: {y.min():.2f}s to {y.max():.2f}s")

    return X, y


# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================

def train_ml_model(X, y, test_size=None):
    """Train ML model with deadline awareness and compare multiple algorithms"""
    if test_size is None:
        test_size = MODEL_CONFIG['test_size']

    print("\nTraining deadline-aware model...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=MODEL_CONFIG['random_state']
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Scale features
    X_train_scaled = model_state['scaler'].fit_transform(X_train)
    X_test_scaled = model_state['scaler'].transform(X_test)

    # Train and compare multiple models
    print("\nComparing models:")

    best_model = None
    best_score = float('-inf')
    best_name = None

    for name, config in MODEL_ALGORITHMS.items():
        # Initialize and train model
        model = config['class'](**config['params'])
        model.fit(X_train_scaled, y_train)

        # Evaluate performance
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        print(f"{name}:")
        print(f"  Train R²: {train_r2:.3f}")
        print(f"  Test R²: {test_r2:.3f}")
        print(f"  Test RMSE: {test_rmse:.3f}s")

        # Select best model based on test R²
        if test_r2 > best_score:
            best_score = test_r2
            best_model = model
            best_name = name

    print(f"\nSelected model: {best_name}")
    model_state['trained_model'] = best_model

    # Calculate final metrics with best model
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\nFinal Model Performance:")
    print(f"Training RMSE: {train_rmse:.3f}s")
    print(f"Test RMSE: {test_rmse:.3f}s")
    print(f"Training MAE: {train_mae:.3f}s")
    print(f"Test MAE: {test_mae:.3f}s")
    print(f"Training R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")

    # Store results for analysis
    results = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_train_pred': y_train_pred, 'y_test_pred': y_test_pred,
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'test_r2': test_r2,
        'model_name': best_name
    }

    return results


def analyze_feature_importance():
    """Analyze which features are most important including deadline constraint"""
    if model_state['trained_model'] is None:
        print("Model not trained yet")
        return None

    print("\nFeature Importance Analysis:")

    # Get feature importance
    if hasattr(model_state['trained_model'], 'feature_importances_'):
        importance = model_state['trained_model'].feature_importances_
    else:
        print("Model does not have feature importance")
        return None

    # Create DataFrame for easy sorting
    importance_df = pd.DataFrame({
        'feature': model_state['feature_names'],
        'importance': importance
    }).sort_values('importance', ascending=False)

    print(importance_df.to_string(index=False))

    # Highlight deadline constraint importance if present
    if 'deadline_constraint' in model_state['feature_names']:
        deadline_importance = importance_df[importance_df['feature'] == 'deadline_constraint']['importance'].iloc[0]
        print(f"\nDeadline constraint importance: {deadline_importance:.1%}")
        print("This shows how much the model considers deadline pressure in its predictions.")

    return importance_df


# =============================================================================
# PREDICTION AND OPTIMIZATION
# =============================================================================

def predict_algorithm_runtime(graph_features, num_nodes, algorithm_type='PageRank', deadline_constraint=60):
    """Predict runtime for given configuration using trained model"""
    if model_state['trained_model'] is None:
        print("Model not trained yet")
        return None

    # Create features dictionary
    features_dict = {
        'nodes': graph_features['nodes'],
        'edges': graph_features['edges'],
        'density': graph_features['density'],
        'avg_clustering': graph_features['avg_clustering'],
        'num_nodes': num_nodes,
        'total_cores': num_nodes * MODEL_CONFIG['cores_per_node'],
        'total_memory_gb': num_nodes * MODEL_CONFIG['memory_per_node_gb']
    }

    # Add deadline constraint if model supports it
    if 'deadline_constraint' in model_state['feature_names']:
        features_dict['deadline_constraint'] = deadline_constraint

    # Add encoded algorithm type if model supports it
    if 'algorithm_type_encoded' in model_state['feature_names']:
        if algorithm_type in model_state['label_encoder'].classes_:
            features_dict['algorithm_type_encoded'] = model_state['label_encoder'].transform([algorithm_type])[0]
        else:
            features_dict['algorithm_type_encoded'] = 0

    # Create feature vector in correct order
    X = pd.DataFrame([features_dict])
    X = X[model_state['feature_names']]

    # Scale features and make prediction
    X_scaled = model_state['scaler'].transform(X)
    prediction = model_state['trained_model'].predict(X_scaled)[0]

    # Ensure minimum realistic time
    return max(prediction, VALIDATION_THRESHOLDS['minimum_realistic_time'])


def optimize_resource_allocation_with_deadline(graph_features, algorithm_type='PageRank', deadline=60.0,
                                               max_nodes=None):
    """Find optimal resource allocation to meet deadline constraint"""
    if max_nodes is None:
        max_nodes = MODEL_CONFIG['max_nodes']

    print(f"\nOptimizing resource allocation for {algorithm_type} with deadline: {deadline}s")

    best_config = None

    # Try different node configurations from 1 to max_nodes
    for num_nodes in range(1, max_nodes + 1):
        predicted_time = predict_algorithm_runtime(graph_features, num_nodes, algorithm_type, deadline)

        print(f"  {num_nodes} nodes → {predicted_time:.2f}s prediction")

        # Check if this configuration meets the deadline
        if predicted_time <= deadline:
            best_config = {
                'num_nodes': num_nodes,
                'predicted_time': predicted_time,
                'algorithm_type': algorithm_type,
                'deadline_constraint': deadline,
                'total_cores': num_nodes * MODEL_CONFIG['cores_per_node'],
                'total_memory_gb': num_nodes * MODEL_CONFIG['memory_per_node_gb'],
                'cost': num_nodes,
                'deadline_margin': deadline - predicted_time
            }
            break

    # Display results
    if best_config:
        print(f"Optimal configuration: {best_config['num_nodes']} nodes")
        print(f"  Predicted time: {best_config['predicted_time']:.2f}s")
        print(f"  Algorithm: {best_config['algorithm_type']}")
        print(f"  Deadline: {best_config['deadline_constraint']}s")
        print(f"  Margin: {best_config['deadline_margin']:.1f}s")
        print(f"  Cost: {best_config['cost']} nodes")
    else:
        print(f"No configuration found meeting {deadline}s deadline")

    return best_config


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_model_performance_visualizations(results, output_dir=None):
    """Create comprehensive visualizations for model performance"""
    if output_dir is None:
        output_dir = DEFAULT_PATHS['visualizations']

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(20, 24))

    # 1. Actual vs Predicted
    axes[0, 0].scatter(results['y_test'], results['y_test_pred'], alpha=0.7, color='blue')
    axes[0, 0].plot([results['y_test'].min(), results['y_test'].max()],
                    [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Runtime (s)')
    axes[0, 0].set_ylabel('Predicted Runtime (s)')
    axes[0, 0].set_title(f'Test Set: Actual vs Predicted\nR² = {results["test_r2"]:.3f}')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Residuals Plot
    residuals = results['y_test'] - results['y_test_pred']
    axes[0, 1].scatter(results['y_test_pred'], residuals, alpha=0.7, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Runtime (s)')
    axes[0, 1].set_ylabel('Residuals (s)')
    axes[0, 1].set_title('Residuals Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Feature Importance
    importance_df = analyze_feature_importance()
    if importance_df is not None:
        axes[1, 0].barh(importance_df['feature'], importance_df['importance'])
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Feature Importance (Including Deadline)')
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Performance Metrics Comparison
    metrics = ['RMSE', 'MAE', 'R²']
    train_values = [results['train_rmse'], results['train_mae'], results['train_r2']]
    test_values = [results['test_rmse'], results['test_mae'], results['test_r2']]

    x = np.arange(len(metrics))
    width = 0.35

    axes[1, 1].bar(x - width / 2, train_values, width, label='Training', alpha=0.8, color='lightblue')
    axes[1, 1].bar(x + width / 2, test_values, width, label='Test', alpha=0.8, color='lightcoral')
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Values')
    axes[1, 1].set_title(f'Model Performance ({results["model_name"]})')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save visualization
    viz_path = f"{output_dir}/model_performance.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Visualizations saved to {viz_path}")


# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

def validate_model_against_baseline(df):
    """Validate model predictions against baseline results with deadline awareness"""
    print("\nValidating model against baseline results...")

    validation_results = []

    for _, row in df.iterrows():
        graph_features = {
            'nodes': row['nodes'],
            'edges': row['edges'],
            'density': row['density'],
            'avg_clustering': row['avg_clustering']
        }

        algorithm_type = row.get('algorithm_type', 'PageRank')
        deadline_constraint = row.get('deadline_constraint', VALIDATION_THRESHOLDS['default_deadline'])

        predicted_time = predict_algorithm_runtime(graph_features, row['num_nodes'], algorithm_type,
                                                   deadline_constraint)
        actual_time = row[MODEL_CONFIG['target_column']]
        error = abs(predicted_time - actual_time)
        error_pct = (error / actual_time) * 100

        validation_results.append({
            'graph_name': row['graph_name'],
            'algorithm_type': algorithm_type,
            'deadline_constraint': deadline_constraint,
            'num_nodes': row['num_nodes'],
            'actual_time': actual_time,
            'predicted_time': predicted_time,
            'error': error,
            'error_pct': error_pct,
            'meets_deadline_actual': row.get('meets_deadline', actual_time <= deadline_constraint),
            'meets_deadline_predicted': predicted_time <= deadline_constraint
        })

    validation_df = pd.DataFrame(validation_results)

    # Overall validation statistics
    print(f"Validation Results:")
    print(f"  Mean absolute error: {validation_df['error'].mean():.3f}s")
    print(f"  Mean percentage error: {validation_df['error_pct'].mean():.1f}%")
    print(f"  Max percentage error: {validation_df['error_pct'].max():.1f}%")

    threshold = VALIDATION_THRESHOLDS['error_percentage_threshold']
    accurate_predictions = (validation_df['error_pct'] <= threshold).sum()
    total_predictions = len(validation_df)
    print(f"  Predictions within {threshold}% error: {accurate_predictions}/{total_predictions}")

    # Algorithm-specific validation
    print(f"\nValidation by Algorithm:")
    for algo in validation_df['algorithm_type'].unique():
        algo_data = validation_df[validation_df['algorithm_type'] == algo]
        mae = algo_data['error'].mean()
        mape = algo_data['error_pct'].mean()
        accurate = (algo_data['error_pct'] <= threshold).sum()
        total = len(algo_data)
        print(f"  {algo}: MAE = {mae:.3f}s, MAPE = {mape:.1f}%, Accuracy = {accurate}/{total}")

    # Deadline-specific validation
    if 'deadline_constraint' in validation_df.columns:
        print(f"\nValidation by Deadline:")
        for deadline in sorted(validation_df['deadline_constraint'].unique()):
            deadline_data = validation_df[validation_df['deadline_constraint'] == deadline]
            mae = deadline_data['error'].mean()
            mape = deadline_data['error_pct'].mean()
            accurate = (deadline_data['error_pct'] <= threshold).sum()
            total = len(deadline_data)
            print(f"  {deadline}s deadline: MAE = {mae:.3f}s, MAPE = {mape:.1f}%, Accuracy = {accurate}/{total}")

    return validation_df


def test_cross_deadline_generalization(df):
    """Test model generalization across different deadline constraints"""
    print("\nTesting Cross-Deadline Generalization...")

    if 'deadline_constraint' not in df.columns:
        print("No deadline constraints found in dataset")
        return None

    deadlines = sorted(df['deadline_constraint'].unique())
    print(f"Testing generalization across deadlines: {deadlines}")

    # Prepare feature columns
    feature_columns = [col for col in model_state['feature_names'] if col != 'deadline_constraint']
    if 'algorithm_type_encoded' not in feature_columns:
        feature_columns.append('algorithm_type_encoded')

    results = []

    # Prepare algorithm encoding
    temp_encoder = LabelEncoder()
    df_encoded = df.copy()
    if 'algorithm_type' in df.columns:
        df_encoded['algorithm_type_encoded'] = temp_encoder.fit_transform(df['algorithm_type'])

    print("Cross-deadline generalization test:")
    print("Training on one deadline, testing on another...\n")

    # Test all deadline combinations
    for train_deadline in deadlines:
        for test_deadline in deadlines:
            if train_deadline != test_deadline:
                # Prepare train/test data by deadline
                train_data = df_encoded[df_encoded['deadline_constraint'] == train_deadline]
                test_data = df_encoded[df_encoded['deadline_constraint'] == test_deadline]

                if len(train_data) > 0 and len(test_data) > 0:
                    X_train = train_data[feature_columns + ['deadline_constraint']]
                    y_train = train_data[MODEL_CONFIG['target_column']]
                    X_test = test_data[feature_columns + ['deadline_constraint']]
                    y_test = test_data[MODEL_CONFIG['target_column']]

                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Train model for this deadline combination
                    model = xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=4,
                        learning_rate=0.1,
                        random_state=42,
                        verbosity=0
                    )
                    model.fit(X_train_scaled, y_train)

                    # Predict and evaluate
                    y_pred = model.predict(X_test_scaled)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                    results.append({
                        'train_deadline': train_deadline,
                        'test_deadline': test_deadline,
                        'test_r2': r2,
                        'test_mae': mae,
                        'test_rmse': rmse,
                        'train_samples': len(X_train),
                        'test_samples': len(X_test)
                    })

                    print(f"Train: {train_deadline}s → Test: {test_deadline}s")
                    print(f"  R²: {r2:.3f}, MAE: {mae:.3f}s, RMSE: {rmse:.3f}s")
                    print(f"  Samples: {len(X_train)} train, {len(X_test)} test\n")

    if results:
        results_df = pd.DataFrame(results)

        # Summary statistics
        print("Cross-Deadline Generalization Summary:")
        avg_r2 = results_df['test_r2'].mean()
        std_r2 = results_df['test_r2'].std()
        avg_mae = results_df['test_mae'].mean()
        std_mae = results_df['test_mae'].std()

        print(f"Average R²: {avg_r2:.3f} ± {std_r2:.3f}")
        print(f"Average MAE: {avg_mae:.3f} ± {std_mae:.3f}s")

        best_idx = results_df['test_r2'].idxmax()
        worst_idx = results_df['test_r2'].idxmin()

        best_result = results_df.loc[best_idx]
        worst_result = results_df.loc[worst_idx]

        print(
            f"Best generalization: {best_result['train_deadline']}s → {best_result['test_deadline']}s (R² = {best_result['test_r2']:.3f})")
        print(
            f"Worst generalization: {worst_result['train_deadline']}s → {worst_result['test_deadline']}s (R² = {worst_result['test_r2']:.3f})")

        return results_df
    else:
        print("No cross deadline pairs found with sufficient data.")
        return None


def compare_with_baseline_schedulers(df):
    """Compare ML model predictions vs baseline schedulers with deadline awareness"""
    print("\n" + "=" * 60)
    print("SCHEDULER COMPARISON ANALYSIS WITH DEADLINE CONSTRAINTS")
    print("=" * 60)

    # Import baseline schedulers
    try:
        import sys
        import os
        sys.path.append(os.getcwd())
        from baseline_schedulers import compare_all_schedulers, estimate_runtime_for_allocation
    except ImportError:
        print("Error: baseline_schedulers.py not found!")
        print("Please ensure baseline_schedulers.py is in the current directory.")
        return None

    comparison_results = []

    print("Comparing ML model vs baseline schedulers across multiple deadlines...")
    print("Testing on subset of experiments for efficiency...\n")

    # Test on representative subset for efficiency
    test_subset = df.iloc[::12].copy()  # Every 12th row for multiple deadlines

    for _, row in test_subset.iterrows():
        graph_features = {
            'nodes': row['nodes'],
            'edges': row['edges'],
            'density': row['density'],
            'avg_clustering': row['avg_clustering']
        }

        algorithm_type = row.get('algorithm_type', 'PageRank')
        deadline_constraint = row.get('deadline_constraint', VALIDATION_THRESHOLDS['default_deadline'])
        actual_runtime = row[MODEL_CONFIG['target_column']]
        actual_nodes = row['num_nodes']

        # ML Model Prediction with deadline awareness
        ml_prediction = optimize_resource_allocation_with_deadline(
            graph_features, algorithm_type, deadline=deadline_constraint, max_nodes=8
        )

        if ml_prediction:
            ml_nodes = ml_prediction['num_nodes']
            ml_predicted_time = ml_prediction['predicted_time']
            ml_cost = ml_prediction['cost']
        else:
            ml_nodes = 8  # Fallback to max
            ml_predicted_time = deadline_constraint
            ml_cost = 8

        # Baseline Scheduler Predictions
        scheduler_results = compare_all_schedulers(graph_features, algorithm_type)

        # Collect results for all schedulers
        result_entry = {
            'graph_name': row['graph_name'],
            'algorithm_type': algorithm_type,
            'deadline_constraint': deadline_constraint,
            'actual_runtime': actual_runtime,
            'actual_nodes': actual_nodes,
            'ml_nodes': ml_nodes,
            'ml_predicted_time': ml_predicted_time,
            'ml_cost': ml_cost
        }

        # Add baseline scheduler results
        for scheduler_name, scheduler_result in scheduler_results.items():
            estimated_runtime = estimate_runtime_for_allocation(scheduler_result, graph_features, algorithm_type)
            result_entry[f'{scheduler_name}_nodes'] = scheduler_result['num_nodes']
            result_entry[f'{scheduler_name}_estimated_time'] = estimated_runtime
            result_entry[f'{scheduler_name}_cost'] = scheduler_result['cost']

        comparison_results.append(result_entry)

    # Convert to DataFrame for analysis
    comparison_df = pd.DataFrame(comparison_results)

    # Calculate scheduler performance metrics
    schedulers_list = ['ml', 'yarn', 'spark', 'linear', 'fixed', 'optimal']
    scheduler_performance = {}

    print("Scheduler Performance Comparison Across All Deadlines:")
    print("-" * 60)

    for scheduler in schedulers_list:
        if f'{scheduler}_nodes' in comparison_df.columns:
            # Calculate metrics
            avg_nodes = comparison_df[f'{scheduler}_nodes'].mean()
            avg_cost = comparison_df[f'{scheduler}_cost'].mean()

            # Accuracy (how often predicted allocation meets deadline)
            if scheduler == 'ml':
                estimated_times = comparison_df[f'{scheduler}_predicted_time']
            else:
                estimated_times = comparison_df[f'{scheduler}_estimated_time']

            meets_deadline = (estimated_times <= comparison_df['deadline_constraint']).sum()
            accuracy = meets_deadline / len(comparison_df) * 100

            # Cost efficiency
            min_possible_cost = comparison_df['optimal_cost'].mean()
            cost_efficiency = (min_possible_cost / avg_cost) * 100 if avg_cost > 0 else 0

            scheduler_performance[scheduler] = {
                'accuracy': accuracy,
                'avg_nodes': avg_nodes,
                'avg_cost': avg_cost,
                'cost_efficiency': cost_efficiency
            }

            scheduler_display_name = {
                'ml': 'ML Model',
                'yarn': 'YARN Fair',
                'spark': 'Spark Default',
                'linear': 'Linear Scaling',
                'fixed': 'Fixed (4 nodes)',
                'optimal': 'Optimal Oracle'
            }.get(scheduler, scheduler)

            print(f"{scheduler_display_name:15}: {accuracy:5.1f}% accuracy, "
                  f"{avg_nodes:.1f} avg nodes, {cost_efficiency:5.1f}% cost efficiency")

    # Deadline-specific analysis
    if 'deadline_constraint' in comparison_df.columns:
        print(f"\nPerformance by Deadline Constraint:")
        print("-" * 50)

        for deadline in sorted(comparison_df['deadline_constraint'].unique()):
            deadline_data = comparison_df[comparison_df['deadline_constraint'] == deadline]
            print(f"\nDeadline {deadline}s:")

            for scheduler in ['ml', 'yarn', 'spark']:
                if f'{scheduler}_nodes' in deadline_data.columns:
                    if scheduler == 'ml':
                        estimated_times = deadline_data[f'{scheduler}_predicted_time']
                    else:
                        estimated_times = deadline_data[f'{scheduler}_estimated_time']

                    meets_deadline = (estimated_times <= deadline).sum()
                    accuracy = meets_deadline / len(deadline_data) * 100
                    avg_nodes = deadline_data[f'{scheduler}_nodes'].mean()

                    scheduler_name = {'ml': 'ML Model', 'yarn': 'YARN', 'spark': 'Spark'}.get(scheduler, scheduler)
                    print(f"  {scheduler_name:10}: {accuracy:5.1f}% success, {avg_nodes:.1f} avg nodes")

    # Overall summary
    print("\n" + "=" * 60)
    print("SCHEDULER COMPARISON SUMMARY")
    print("=" * 60)

    # Find best scheduler for each metric
    best_accuracy = max(scheduler_performance.keys(),
                        key=lambda x: scheduler_performance[x]['accuracy'])
    best_cost_efficiency = max(scheduler_performance.keys(),
                               key=lambda x: scheduler_performance[x]['cost_efficiency'])
    lowest_cost = min(scheduler_performance.keys(),
                      key=lambda x: scheduler_performance[x]['avg_cost'])

    print(f"Best accuracy: {best_accuracy.upper()} ({scheduler_performance[best_accuracy]['accuracy']:.1f}%)")
    print(
        f"Best cost efficiency: {best_cost_efficiency.upper()} ({scheduler_performance[best_cost_efficiency]['cost_efficiency']:.1f}%)")
    print(f"Lowest average cost: {lowest_cost.upper()} ({scheduler_performance[lowest_cost]['avg_cost']:.1f} nodes)")

    # ML Model analysis
    if 'ml' in scheduler_performance:
        ml_perf = scheduler_performance['ml']
        print(f"\nML Model Performance:")
        print(f"  Deadline satisfaction: {ml_perf['accuracy']:.1f}%")
        print(f"  Average resource usage: {ml_perf['avg_nodes']:.1f} nodes")
        print(f"  Cost efficiency: {ml_perf['cost_efficiency']:.1f}%")

    # Save detailed comparison results
    output_dir = Path(DEFAULT_PATHS['scheduler_comparison'])
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_dir / "scheduler_comparison_results.csv", index=False)

    print(f"\nDetailed comparison results saved to: {output_dir}/scheduler_comparison_results.csv")
    print("=" * 60)

    return comparison_df


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_trained_model(filepath=None):
    """Save trained deadline-aware model to disk"""
    if filepath is None:
        filepath = DEFAULT_PATHS['model_save']

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'model': model_state['trained_model'],
        'scaler': model_state['scaler'],
        'label_encoder': model_state['label_encoder'],
        'feature_names': model_state['feature_names'],
        'target_name': model_state['target_name']
    }

    joblib.dump(model_data, filepath)
    print(f"Deadline-aware model saved to {filepath}")


def load_trained_model(filepath=None):
    """Load trained deadline-aware model from disk"""
    if filepath is None:
        filepath = DEFAULT_PATHS['model_save']

    try:
        model_data = joblib.load(filepath)
        model_state['trained_model'] = model_data['model']
        model_state['scaler'] = model_data['scaler']
        model_state['label_encoder'] = model_data.get('label_encoder', LabelEncoder())
        model_state['feature_names'] = model_data['feature_names']
        model_state['target_name'] = model_data['target_name']
        print(f"Deadline-aware model loaded from {filepath}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

def test_model_on_multiple_deadlines():
    """Test the trained model on hypothetical graphs with different deadline constraints"""
    print("\nTesting Deadline-Aware Model on New Graph with Multiple Deadlines...")

    # Create hypothetical new graph features
    new_graph_features = {
        'nodes': 50000,
        'edges': 150000,
        'density': 0.00012,
        'avg_clustering': 0.03
    }

    print(f"New graph: {new_graph_features['nodes']:,} nodes, {new_graph_features['edges']:,} edges")

    # Load model if not already trained
    if model_state['trained_model'] is None:
        if not load_trained_model():
            print("No trained model available")
            return False

    # Test all algorithms with different deadlines
    algorithms = ['PageRank', 'ConnectedComponents', 'TriangleCounting']
    deadlines = [20, 30, 45, 60]

    print("\nOptimal Resource Allocation by Algorithm and Deadline:")
    print("-" * 60)

    for algorithm in algorithms:
        print(f"\n{algorithm}:")
        for deadline in deadlines:
            optimal_config = optimize_resource_allocation_with_deadline(
                new_graph_features, algorithm_type=algorithm, deadline=deadline
            )

            if optimal_config:
                predicted_time = optimal_config['predicted_time']
                margin = optimal_config['deadline_margin']
                nodes = optimal_config['num_nodes']
                print(f"  {deadline}s: {nodes} nodes ({predicted_time:.1f}s predicted, {margin:.1f}s margin)")
            else:
                print(f"  {deadline}s: No solution found within 8 nodes")

    return True


def demonstrate_model_usage():
    """Demonstrate how to use the trained model with example graphs"""
    print("\nTesting Model on Dataset Examples with Multiple Deadlines:")

    test_graphs = [
        {'name': 'p2p-Gnutella06', 'nodes': 8717, 'edges': 31525, 'density': 0.000830, 'avg_clustering': 0.0067},
        {'name': 'ca-AstroPh', 'nodes': 18772, 'edges': 198110, 'density': 0.001124, 'avg_clustering': 0.6306},
        {'name': 'email-EuAll', 'nodes': 265214, 'edges': 365570, 'density': 0.000010, 'avg_clustering': 0.2379}
    ]

    algorithms = ['PageRank', 'ConnectedComponents', 'TriangleCounting']
    deadlines = [20, 30, 45, 60]

    # Test each graph with different algorithms and deadlines
    for graph in test_graphs:
        print(f"\nTesting {graph['name']}:")
        for algorithm in algorithms:
            print(f"  {algorithm}:")
            for deadline in deadlines:
                optimal_config = optimize_resource_allocation_with_deadline(
                    graph, algorithm, deadline=deadline
                )
                if optimal_config:
                    print(f"    {deadline}s: {optimal_config['num_nodes']} nodes")
                else:
                    print(f"    {deadline}s: No solution")


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """Main training and testing pipeline for deadline-aware ML model"""
    print("ML Model Training & Testing with Multiple Deadline Constraints\n")

    # Load baseline experiment data
    df = load_baseline_experiment_data()
    if df is None:
        print("Cannot proceed without baseline data")
        return

    # Prepare features including deadline constraints
    X, y = prepare_ml_features(df)

    # Train deadline-aware model
    results = train_ml_model(X, y)

    # Analyze feature importance including deadline
    importance_df = analyze_feature_importance()

    # Create performance visualizations
    create_model_performance_visualizations(results)

    # Validate against baseline with deadline awareness
    validation_results = validate_model_against_baseline(df)

    # Save deadline-aware model
    save_trained_model()

    # Test cross-deadline generalization
    print("\n" + "=" * 60)
    generalization_results = test_cross_deadline_generalization(df)
    print("=" * 60)

    # Compare with baseline schedulers across deadlines
    scheduler_comparison = compare_with_baseline_schedulers(df)

    # Demonstrate model usage
    demonstrate_model_usage()

    # Test on hypothetical new graph with multiple deadlines
    test_model_on_multiple_deadlines()

    # Final summary
    print("\nDeadline-Aware Model Training Complete!")
    print(f"Model trained on {len(df)} experiments from {df['graph_name'].nunique()} datasets")
    print(f"Algorithms included: {df['algorithm_type'].unique()}")

    if 'deadline_constraint' in df.columns:
        deadlines = sorted(df['deadline_constraint'].unique())
        print(f"Deadline constraints: {deadlines} seconds")

    print("\nModel Performance Summary:")
    print(f"  Test R²: {results['test_r2']:.3f}")
    print(f"  Test RMSE: {results['test_rmse']:.3f}s")
    print(f"  Selected algorithm: {results['model_name']}")

    # Enhanced deadline-aware features
    if importance_df is not None and 'deadline_constraint' in importance_df['feature'].values:
        deadline_importance = importance_df[importance_df['feature'] == 'deadline_constraint']['importance'].iloc[0]
        print(f"  Deadline constraint importance: {deadline_importance:.1%}")

    print("\nChallenging Scenarios Successfully Addressed:")
    if 'deadline_constraint' in df.columns:
        challenging_count = 0
        for deadline in [20, 30]:  # Tight deadlines
            deadline_data = df[df['deadline_constraint'] == deadline]
            failures = deadline_data[deadline_data['meets_deadline'] == False]
            challenging_count += len(failures)
        print(f" {challenging_count} challenging scenarios where resource allocation is critical")
        print(" ML model can predict when additional resources are needed")


if __name__ == "__main__":
    main()