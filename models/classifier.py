import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple, Any
import config

def create_rf_pipeline() -> Pipeline:
    """
    Create Random Forest classification pipeline with preprocessing.
    
    Returns:
    --------
    Pipeline
        Sklearn pipeline with StandardScaler and RandomForestClassifier
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            random_state=config.MODEL_PARAMS['random_state'], 
            class_weight='balanced'
        ))
    ])
    
    return pipeline

def perform_grid_search(X: np.ndarray, y: np.ndarray, 
                       pipeline: Pipeline = None) -> GridSearchCV:
    """
    Perform hyperparameter optimization using Grid Search with Cross-Validation.
    
    This function prevents data leakage by only using training+validation data
    for hyperparameter tuning, keeping the test set completely unseen.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (training + validation data only)
    y : np.ndarray
        Labels (training + validation data only)
    pipeline : Pipeline, optional
        Classification pipeline. Creates default if None
        
    Returns:
    --------
    GridSearchCV
        Fitted grid search object
    """
    if pipeline is None:
        pipeline = create_rf_pipeline()
    
    # Setup cross-validation
    cv = StratifiedKFold(
        n_splits=config.MODEL_PARAMS['cv_folds'], 
        shuffle=True, 
        random_state=config.MODEL_PARAMS['random_state']
    )
    
    # Create grid search
    grid_search = GridSearchCV(
        pipeline, 
        config.RF_PARAM_GRID,
        cv=cv,
        scoring='accuracy', 
        n_jobs=config.MODEL_PARAMS['n_jobs'], 
        verbose=1
    )
    
    print(f"\n🔄 Running GridSearchCV on Train+Validation set...")
    param_combinations = 1
    for param_values in config.RF_PARAM_GRID.values():
        param_combinations *= len(param_values)
    
    print(f"   • Searching {param_combinations} parameter combinations")
    print(f"   • Using {config.MODEL_PARAMS['cv_folds']}-fold cross-validation")
    
    # Fit grid search
    grid_search.fit(X, y)
    
    return grid_search

def evaluate_on_test_set(grid_search: GridSearchCV, X_test: np.ndarray, 
                        y_test: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate the best model on the held-out test set.
    
    Parameters:
    -----------
    grid_search : GridSearchCV
        Fitted grid search object
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
        
    Returns:
    --------
    dict
        Dictionary containing evaluation results
    """
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Make predictions on test set
    test_predictions = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    # Get detailed metrics
    target_names = ['Tongue', 'Hand']
    class_report = classification_report(
        y_test, test_predictions, 
        target_names=target_names, 
        zero_division=0,
        output_dict=True
    )
    
    confusion_mat = confusion_matrix(y_test, test_predictions)
    
    # Feature importance
    rf_model = best_model.named_steps['rf']
    feature_importance = rf_model.feature_importances_
    
    # Compile results
    results = {
        'cv_score': grid_search.best_score_,
        'test_accuracy': test_accuracy,
        'best_params': grid_search.best_params_,
        'predictions': test_predictions,
        'classification_report': class_report,
        'confusion_matrix': confusion_mat,
        'feature_importance': feature_importance,
        'model': best_model
    }
    
    return results

def perform_nested_cv(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Perform nested cross-validation for robust performance estimation.
    
    Nested CV provides an unbiased estimate of model performance by using
    separate CV loops for hyperparameter tuning and performance evaluation.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (training + validation data)
    y : np.ndarray
        Labels (training + validation data)
        
    Returns:
    --------
    dict
        Nested CV results
    """
    print(f"\n🔄 Running Nested Cross-Validation...")
    
    # Outer CV for performance estimation
    outer_cv = StratifiedKFold(
        n_splits=config.MODEL_PARAMS['cv_folds'], 
        shuffle=True, 
        random_state=123
    )
    
    nested_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y)):
        print(f"   Processing outer fold {fold_idx+1}/{config.MODEL_PARAMS['cv_folds']}...")
        
        # Split data for this outer fold
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Inner GridSearch (only on training data of this fold)
        pipeline = create_rf_pipeline()
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        inner_grid = GridSearchCV(
            pipeline, 
            config.NESTED_RF_PARAM_GRID, 
            cv=inner_cv, 
            scoring='accuracy', 
            n_jobs=config.MODEL_PARAMS['n_jobs'], 
            verbose=0
        )
        
        inner_grid.fit(X_train_fold, y_train_fold)
        
        # Evaluate best model on validation fold
        fold_score = inner_grid.score(X_val_fold, y_val_fold)
        nested_scores.append(fold_score)
    
    nested_results = {
        'mean_score': np.mean(nested_scores),
        'std_score': np.std(nested_scores),
        'individual_scores': nested_scores
    }
    
    return nested_results

def print_classification_results(results: Dict[str, Any], feature_names: list = None) -> None:
    """
    Print formatted classification results.
    
    Parameters:
    -----------
    results : dict
        Results from evaluate_on_test_set()
    feature_names : list, optional
        Names of features for importance display
    """
    print("\n" + "="*60)
    print("              CLASSIFICATION RESULTS")
    print("="*60)
    
    # Best hyperparameters
    print(f"🏆 Best Cross-Validation Score: {results['cv_score']:.3f}")
    print(f"\n🔥 Best Hyperparameters:")
    for param, value in results['best_params'].items():
        param_name = param.replace('rf__', '')
        print(f"   • {param_name}: {value}")
    
    # Test set performance
    print(f"\n🧪 Final Evaluation on Test Set:")
    print(f"   • Test Accuracy: {results['test_accuracy']:.3f}")
    print(f"   • This is the TRUE performance estimate!")
    
    # Feature importance
    if feature_names is not None and len(feature_names) == len(results['feature_importance']):
        print(f"\n🔍 Feature Importance:")
        # Sort features by importance
        importance_pairs = list(zip(feature_names, results['feature_importance']))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for name, importance in importance_pairs:
            print(f"   • {name}: {importance:.4f}")
    else:
        print(f"\n🔍 Feature Importance:")
        for i, importance in enumerate(results['feature_importance']):
            print(f"   • Feature {i+1}: {importance:.4f}")
    
    # Classification report
    print(f"\n📋 Detailed Test Set Results:")
    print("Classification Report:")
    class_report = results['classification_report']
    print(f"   • Tongue: precision={class_report['Tongue']['precision']:.3f}, "
          f"recall={class_report['Tongue']['recall']:.3f}, "
          f"f1={class_report['Tongue']['f1-score']:.3f}")
    print(f"   • Hand:   precision={class_report['Hand']['precision']:.3f}, "
          f"recall={class_report['Hand']['recall']:.3f}, "
          f"f1={class_report['Hand']['f1-score']:.3f}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print(f"   • True Tongue, Pred Tongue: {cm[0,0]}")
    print(f"   • True Tongue, Pred Hand:   {cm[0,1]}")
    print(f"   • True Hand,   Pred Tongue: {cm[1,0]}")
    print(f"   • True Hand,   Pred Hand:   {cm[1,1]}")

def print_nested_cv_results(nested_results: Dict[str, float]) -> None:
    """
    Print nested cross-validation results.
    
    Parameters:
    -----------
    nested_results : dict
        Results from perform_nested_cv()
    """
    print(f"\n📈 Nested Cross-Validation Results:")
    print(f"   • Mean CV Accuracy: {nested_results['mean_score']:.3f} ± {nested_results['std_score']:.3f}")
    print(f"   • Individual fold scores: {[f'{score:.3f}' for score in nested_results['individual_scores']]}")

def split_data_prevent_leakage(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data to prevent data leakage.
    
    Separates data into training+validation (for hyperparameter tuning) and 
    test set (for final evaluation). The test set is never used during training
    or hyperparameter optimization.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
        
    Returns:
    --------
    tuple
        X_temp, X_test, y_temp, y_test
    """
    print(f"\n🚨 PREVENTING DATA LEAKAGE - Proper Data Splitting...")
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=config.MODEL_PARAMS['test_size'], 
        random_state=config.MODEL_PARAMS['random_state'], 
        stratify=y
    )
    
    print(f"📊 Data Split Summary:")
    print(f"   • Train+Validation: {len(X_temp)} samples")
    print(f"   • Test (held-out): {len(X_test)} samples")
    print(f"   • Test set will NEVER be used during training or hyperparameter tuning!")
    
    return X_temp, X_test, y_temp, y_test

def run_complete_classification_pipeline(X: np.ndarray, y: np.ndarray, 
                                        feature_names: list = None) -> Dict[str, Any]:
    """
    Run the complete classification pipeline with proper data splitting.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels  
    feature_names : list, optional
        Feature names for interpretation
        
    Returns:
    --------
    dict
        Complete results dictionary
    """
    # Split data to prevent leakage
    X_temp, X_test, y_temp, y_test = split_data_prevent_leakage(X, y)
    
    # Perform grid search on training+validation data
    grid_search = perform_grid_search(X_temp, y_temp)
    
    # Evaluate on test set
    test_results = evaluate_on_test_set(grid_search, X_test, y_test)
    
    # Perform nested CV for robust evaluation
    nested_results = perform_nested_cv(X_temp, y_temp)
    
    # Print results
    print_classification_results(test_results, feature_names)
    print_nested_cv_results(nested_results)
    
    # Compile all results
    complete_results = {
        **test_results,
        'nested_cv': nested_results,
        'data_split': {
            'train_val_size': len(X_temp),
            'test_size': len(X_test)
        }
    }
    
    # Summary
    print("\n" + "="*60)
    print("                    SUMMARY")
    print("="*60)
    print(f"✅ Data Leakage: PREVENTED")
    print(f"✅ Model: Random Forest")
    print(f"📊 Cross-Validation Score: {test_results['cv_score']:.3f}")
    print(f"🧪 Test Set Accuracy: {test_results['test_accuracy']:.3f}")
    print(f"🔄 Nested CV Accuracy: {nested_results['mean_score']:.3f} ± {nested_results['std_score']:.3f}")
    print(f"📋 The test set accuracy ({test_results['test_accuracy']:.3f}) is the most reliable estimate!")
    print("="*60)
    
    return complete_results