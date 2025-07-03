#!/usr/bin/env python3
"""
Housing Price Prediction using Regression Models
MLOps Assignment 1 - Regression Branch Implementation

This script implements the main regression workflow for comparing
multiple regression models on the Boston Housing dataset.
"""

import argparse
import sys
from utils import (
    load_data,
    preprocess_data,
    get_regression_models,
    get_hyperparameter_grids,
    train_model,
    train_model_with_hyperparameter_tuning,
    evaluate_model,
    save_model,
    save_results,
    print_results,
    create_performance_summary
)

def run_basic_regression():
    """
    Run basic regression without hyperparameter tuning
    """
    print("Starting Basic Regression Analysis...")
    print("="*50)
    
    # Load and preprocess data
    print("Loading dataset...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Get models
    models = get_regression_models()
    results = {}
    
    # Train and evaluate each model
    print("\nTraining and evaluating models...")
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Train model
        trained_model = train_model(model, X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(trained_model, X_test, y_test)
        results[model_name] = metrics
        
        # Save model
        save_model(trained_model, model_name)
        
        print(f"{model_name} - MSE: {metrics['MSE']:.4f}, R²: {metrics['R2']:.4f}")
    
    # Print and save results
    print_results(results)
    save_results(results, 'basic_regression_results.json')
    create_performance_summary(results, 'basic_regression_summary.txt')
    
    return results

def run_hyperparameter_tuning():
    """
    Run regression with hyperparameter tuning
    """
    print("Starting Hyperparameter Tuning Analysis...")
    print("="*50)
    
    # Load and preprocess data
    print("Loading dataset...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Get models and parameter grids
    models = get_regression_models()
    param_grids = get_hyperparameter_grids()
    results = {}
    best_params = {}
    
    # Train and evaluate each model with hyperparameter tuning
    print("\nTraining models with hyperparameter tuning...")
    for model_name, model in models.items():
        print(f"\nTuning {model_name}...")
        
        # Get parameter grid
        param_grid = param_grids[model_name]
        print(f"Parameter grid: {param_grid}")
        
        # Train with hyperparameter tuning
        best_model, best_model_params = train_model_with_hyperparameter_tuning(
            model, param_grid, X_train, y_train
        )
        
        # Store best parameters
        best_params[model_name] = best_model_params
        
        # Evaluate model
        metrics = evaluate_model(best_model, X_test, y_test)
        results[model_name] = metrics
        
        # Save model
        save_model(best_model, f'{model_name}_tuned')
        
        print(f"Best parameters: {best_model_params}")
        print(f"{model_name} - MSE: {metrics['MSE']:.4f}, R²: {metrics['R2']:.4f}")
    
    # Print and save results
    print_results(results)
    save_results(results, 'hyperparameter_tuning_results.json')
    save_results(best_params, 'best_parameters.json')
    create_performance_summary(results, 'hyperparameter_tuning_summary.txt')
    
    return results, best_params

def main():
    """
    Main function to run the regression analysis
    """
    parser = argparse.ArgumentParser(description='Housing Price Prediction')
    parser.add_argument(
        '--mode', 
        choices=['basic', 'hyperparameter', 'both'], 
        default='both',
        help='Mode to run: basic regression, hyperparameter tuning, or both'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode in ['basic', 'both']:
            print("RUNNING BASIC REGRESSION")
            print("="*60)
            basic_results = run_basic_regression()
            
        if args.mode in ['hyperparameter', 'both']:
            print("\n\nRUNNING HYPERPARAMETER TUNING")
            print("="*60)
            tuned_results, best_params = run_hyperparameter_tuning()
            
        print("\n\nANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Check the generated files:")
        print("- Models saved in 'models/' directory")
        print("- Results saved as JSON files")
        print("- Performance summaries as TXT files")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
