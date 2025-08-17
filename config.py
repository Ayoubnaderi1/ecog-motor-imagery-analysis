# Configuration file for Motor Imagery ECoG Analysis

# Data paths
DATA_PATH = r"C:\Users\Asus\Downloads\MOTOR Imagery\motor_imagery.npz"

# Signal processing parameters
FILTER_PARAMS = {
    'lowcut': 4,
    'highcut': 130,
    'order': 4
}

# Connectivity parameters
CONNECTIVITY_PARAMS = {
    'method': 'imcoh',
    'fmin': 70,
    'fmax': 100,
    'baseline_normalization': 'subtraction_baseline'
}

# ROI definitions
COMMON_ROIS = [
    'Brodmann area 4', 'Brodmann area 6', 'Brodmann area 2', 
    'Brodmann area 9', 'Brodmann area 43', 'Brodmann area 45', 'Brodmann area 46'
]

# Task parameters
TASK_PARAMS = {
    'hand_stim_id': 12,
    'tongue_stim_id': 11,
    'epoch_duration_ms': 3000
}

# Feature extraction parameters
IMPORTANT_CONNECTIONS = [
    (0, 1), (0, 2), (1, 2), (0, 3), (1, 3),
    (2, 3), (0, 4), (1, 4), (2, 4)
]

# Model parameters
MODEL_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'n_jobs': -1
}

# Random Forest hyperparameters for grid search
RF_PARAM_GRID = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [3, 5, 10, None],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}

# Nested CV parameters (reduced for computational efficiency)
NESTED_RF_PARAM_GRID = {
    'rf__n_estimators': [50, 100],
    'rf__max_depth': [5, 10, None],
    'rf__min_samples_split': [2, 5]
}