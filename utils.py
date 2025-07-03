import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import joblib
import os
import json
from datetime import datetime

def load_data():
    """
    Load Boston Housing dataset from CMU repository
    Returns: DataFrame with features and target variable
    """
    import pandas as pd
    import numpy as np
    
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    
    # Split data and target
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    
    # Feature names based on original dataset
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target  # Target variable
    
    return df

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the data by splitting and scaling
    
    Args:
        df: Input DataFrame
        test_size: Test split ratio
        random_state: Random state for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Separate features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def get_regression_models():
    """
    Return dictionary of regression models to evaluate
    
    Returns:
        Dictionary of model name and model object
    """
    models = {
        'Linear_Regression': LinearRegression(),
        'Random_Forest': RandomForestRegressor(random_state=42),
        'Support_Vector_Regression': SVR()
    }
    return models

def get_hyperparameter_grids():
    """
    Return hyperparameter grids for each model
    
    Returns:
        Dictionary of model name and parameter grid
    """
    param_grids = {
        'Linear_Regression': {
            'fit_intercept': [True, False],
            'normalize': [True, False],
            'copy_X': [True, False]
        },
        'Random_Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        'Support_Vector_Regression': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001]
        }
    }
    return param_grids

def train_model(model, X_train, y_train):
    """
    Train a single model
    
    Args:
        model: Sklearn model object
        X_train: Training features
        y_train: Training target
    
    Returns:
        Trained model
    """
    model.fit(X_train, y_train)
    return model

def train_model_with_hyperparameter_tuning(model, param_grid, X_train, y_train, cv=5):
    """
    Train model with hyperparameter tuning using GridSearchCV
    
    Args:
        model: Sklearn model object
        param_grid: Parameter grid for tuning
        X_train: Training features
        y_train: Training target
        cv: Cross-validation folds
    
    Returns:
        Best model after hyperparameter tuning
    """
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    
    Returns:
        Dictionary with MSE and R² scores
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'MSE': mse,
        'R2': r2,
        'RMSE': np.sqrt(mse)
    }

def save_model(model, model_name, directory='models'):
    """
    Save trained model to disk
    
    Args:
        model: Trained model
        model_name: Name for the model file
        directory: Directory to save model
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filename = os.path.join(directory, f'{model_name}.pkl')
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def save_results(results, filename='results.json'):
    """
    Save results to JSON file
    
    Args:
        results: Dictionary of results
        filename: Output filename
    """
    results['timestamp'] = datetime.now().isoformat()
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {filename}")

def load_model(model_path):
    """
    Load saved model
    
    Args:
        model_path: Path to saved model
    
    Returns:
        Loaded model
    """
    return joblib.load(model_path)

def print_results(results):
    """
    Print formatted results
    
    Args:
        results: Dictionary of model results
    """
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    
    for model_name, metrics in results.items():
        if model_name != 'timestamp':
            print(f"\n{model_name}:")
            print(f"  MSE:  {metrics['MSE']:.4f}")
            print(f"  R²:   {metrics['R2']:.4f}")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
    
    print("\n" + "="*60)

def get_best_model(results):
    """
    Get the best performing model based on R² score
    
    Args:
        results: Dictionary of model results
    
    Returns:
        Name of best model
    """
    best_model = None
    best_r2 = -float('inf')
    
    for model_name, metrics in results.items():
        if model_name != 'timestamp' and metrics['R2'] > best_r2:
            best_r2 = metrics['R2']
            best_model = model_name
    
    return best_model, best_r2

def create_performance_summary(results, output_file='performance_summary.txt'):
    """
    Create a detailed performance summary report
    
    Args:
        results: Dictionary of model results
        output_file: Output file name
    """
    with open(output_file, 'w') as f:
        f.write("HOUSE PRICE PREDICTION - MODEL PERFORMANCE REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model comparison
        f.write("MODEL PERFORMANCE COMPARISON:\n")
        f.write("-" * 40 + "\n")
        
        for model_name, metrics in results.items():
            if model_name != 'timestamp':
                f.write(f"\n{model_name}:\n")
                f.write(f"  Mean Squared Error (MSE): {metrics['MSE']:.4f}\n")
                f.write(f"  R-squared (R²):          {metrics['R2']:.4f}\n")
                f.write(f"  Root MSE (RMSE):         {metrics['RMSE']:.4f}\n")
        
        # Best model
        best_model, best_r2 = get_best_model(results)
        f.write(f"\nBEST PERFORMING MODEL: {best_model}\n")
        f.write(f"Best R² Score: {best_r2:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"Performance summary saved to {output_file}")
