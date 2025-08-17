import numpy as np
from typing import List, Tuple
import config
from .roi import get_roi_pair_name

def extract_connectivity_features(roi_matrices: np.ndarray, 
                                 important_connections: List[Tuple[int, int]] = None) -> np.ndarray:
    """
    Extract specific connectivity features from ROI matrices.
    
    This function selects a subset of ROI-to-ROI connections that are most
    relevant for the classification task. This reduces dimensionality and
    focuses on the most informative connections.
    
    Parameters:
    -----------
    roi_matrices : np.ndarray
        Array of ROI connectivity matrices (n_trials x n_rois x n_rois)
    important_connections : list, optional
        List of (i, j) tuples specifying which ROI pairs to extract.
        Uses config if None.
        
    Returns:
    --------
    np.ndarray
        Feature matrix (n_trials x n_features)
    """
    if important_connections is None:
        important_connections = config.IMPORTANT_CONNECTIONS
    
    n_trials = len(roi_matrices)
    n_features = len(important_connections)
    
    # Initialize feature matrix
    features = np.zeros((n_trials, n_features))
    
    # Extract features for each trial
    for trial_idx, matrix in enumerate(roi_matrices):
        for feat_idx, (i, j) in enumerate(important_connections):
            # Check if indices are valid for this matrix
            if i < matrix.shape[0] and j < matrix.shape[1]:
                features[trial_idx, feat_idx] = matrix[i, j]
            else:
                # Set to 0 if ROI pair not available
                features[trial_idx, feat_idx] = 0.0
    
    return features

def get_feature_names(important_connections: List[Tuple[int, int]] = None, 
                     short_names: bool = True) -> List[str]:
    """
    Get descriptive names for connectivity features.
    
    Parameters:
    -----------
    important_connections : list, optional
        List of (i, j) tuples. Uses config if None
    short_names : bool
        Use abbreviated ROI names
        
    Returns:
    --------
    list
        List of feature names
    """
    if important_connections is None:
        important_connections = config.IMPORTANT_CONNECTIONS
    
    feature_names = []
    for i, j in important_connections:
        pair_name = get_roi_pair_name(i, j, short_names)
        feature_names.append(pair_name)
    
    return feature_names

def analyze_feature_statistics(features: np.ndarray, labels: np.ndarray = None) -> dict:
    """
    Analyze statistical properties of extracted features.
    
    Parameters:
    -----------
    features : np.ndarray
        Feature matrix (n_trials x n_features)
    labels : np.ndarray, optional
        Class labels for group statistics
        
    Returns:
    --------
    dict
        Dictionary containing feature statistics
    """
    n_trials, n_features = features.shape
    
    # Basic statistics
    stats = {
        'n_trials': n_trials,
        'n_features': n_features,
        'feature_means': np.mean(features, axis=0),
        'feature_stds': np.std(features, axis=0),
        'feature_range': (np.min(features), np.max(features)),
        'zero_variance_features': np.sum(np.std(features, axis=0) == 0)
    }
    
    # Class-specific statistics if labels provided
    if labels is not None:
        unique_labels = np.unique(labels)
        stats['class_means'] = {}
        stats['class_stds'] = {}
        
        for label in unique_labels:
            mask = labels == label
            stats['class_means'][label] = np.mean(features[mask], axis=0)
            stats['class_stds'][label] = np.std(features[mask], axis=0)
    
    # Check for extreme values
    extreme_threshold = 3 * np.std(features)
    stats['n_extreme_values'] = np.sum(np.abs(features) > extreme_threshold)
    stats['extreme_percentage'] = (stats['n_extreme_values'] / features.size) * 100
    
    return stats

def print_feature_statistics(stats: dict, feature_names: List[str] = None) -> None:
    """
    Print feature statistics in a formatted way.
    
    Parameters:
    -----------
    stats : dict
        Output from analyze_feature_statistics()
    feature_names : list, optional
        Feature names for detailed output
    """
    print(f"\n🤖 Feature Preparation:")
    print(f"   • Total trials: {stats['n_trials']}")
    print(f"   • Features per trial: {stats['n_features']}")
    print(f"   • Feature range: [{stats['feature_range'][0]:.4f}, {stats['feature_range'][1]:.4f}]")
    print(f"   • Features with zero variance: {stats['zero_variance_features']}")
    print(f"   • Extreme values (>3σ): {stats['n_extreme_values']} ({stats['extreme_percentage']:.2f}%)")
    
    # Detailed feature statistics if names provided
    if feature_names is not None and len(feature_names) == stats['n_features']:
        print(f"\n🔍 Individual Feature Statistics:")
        for i, name in enumerate(feature_names):
            mean_val = stats['feature_means'][i]
            std_val = stats['feature_stds'][i]
            print(f"   • {name}: mean={mean_val:.4f}, std={std_val:.6f}")

def validate_features(features: np.ndarray, labels: np.ndarray) -> bool:
    """
    Validate feature matrix for potential issues.
    
    Parameters:
    -----------
    features : np.ndarray
        Feature matrix
    labels : np.ndarray
        Class labels
        
    Returns:
    --------
    bool
        True if features pass validation
    """
    issues = []
    
    # Check for NaN values
    if np.isnan(features).any():
        issues.append("Contains NaN values")
    
    # Check for infinite values
    if np.isinf(features).any():
        issues.append("Contains infinite values")
    
    # Check for constant features
    n_constant = np.sum(np.std(features, axis=0) == 0)
    if n_constant > 0:
        issues.append(f"{n_constant} constant features")
    
    # Check for insufficient samples
    if len(features) < 10:
        issues.append("Insufficient samples (<10)")
    
    # Check label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_class_size = np.min(counts)
    if min_class_size < 2:
        issues.append(f"Class with <2 samples: {min_class_size}")
    
    # Print validation results
    if issues:
        print(f"⚠️ Feature validation issues:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    else:
        print(f"✅ Feature validation passed")
        return True

def prepare_features_for_classification(roi_matrices: np.ndarray, 
                                       labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Complete feature preparation pipeline.
    
    Parameters:
    -----------
    roi_matrices : np.ndarray
        ROI connectivity matrices
    labels : np.ndarray
        Class labels
        
    Returns:
    --------
    tuple
        Features, labels, and statistics dictionary
    """
    # Extract features
    features = extract_connectivity_features(roi_matrices)
    
    # Analyze statistics
    stats = analyze_feature_statistics(features, labels)
    
    # Get feature names
    feature_names = get_feature_names()
    
    # Print statistics
    print_feature_statistics(stats, feature_names)
    
    # Validate features
    is_valid = validate_features(features, labels)
    
    # Add validation result to stats
    stats['validation_passed'] = is_valid
    stats['feature_names'] = feature_names
    
    return features, labels, stats