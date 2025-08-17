import numpy as np
from mne_connectivity import spectral_connectivity_epochs
from typing import Tuple
import config

def compute_trial_connectivity(trial_data: np.ndarray, srate: float, 
                              method: str = None, fmin: float = None, fmax: float = None) -> np.ndarray:
    """
    Compute connectivity matrix for a single trial.
    
    Parameters:
    -----------
    trial_data : np.ndarray
        Single trial data (n_channels x n_samples)
    srate : float
        Sampling rate
    method : str, optional
        Connectivity method. Uses config if None
    fmin : float, optional
        Minimum frequency. Uses config if None  
    fmax : float, optional
        Maximum frequency. Uses config if None
        
    Returns:
    --------
    np.ndarray
        Connectivity matrix (n_channels x n_channels)
    """
    # Use config defaults if parameters not provided
    if method is None:
        method = config.CONNECTIVITY_PARAMS['method']
    if fmin is None:
        fmin = config.CONNECTIVITY_PARAMS['fmin']
    if fmax is None:
        fmax = config.CONNECTIVITY_PARAMS['fmax']
    
    # Reshape for MNE connectivity: (n_trials=1, n_channels, n_samples)
    trial_data_reshaped = trial_data[np.newaxis, :, :]
    
    # Compute connectivity
    conn = spectral_connectivity_epochs(
        trial_data_reshaped, 
        method=method, 
        sfreq=srate,
        fmin=fmin, 
        fmax=fmax, 
        faverage=True, 
        verbose=False
    )
    
    # Extract connectivity matrix: (n_channels x n_channels)
    connectivity_matrix = conn.get_data(output='dense')[:, :, 0]
    return connectivity_matrix

def compute_baseline_connectivity(baseline_data: np.ndarray, srate: float, 
                                 method: str = None, fmin: float = None, fmax: float = None) -> np.ndarray:
    """
    Compute connectivity matrix for baseline period.
    
    The baseline connectivity represents the resting-state functional connectivity
    before any task-related activity. This is used for normalization of trial 
    connectivity values.
    
    Parameters:
    -----------
    baseline_data : np.ndarray
        Baseline period data (n_channels x n_samples)
    srate : float
        Sampling rate
    method : str, optional
        Connectivity method. Uses config if None
    fmin : float, optional
        Minimum frequency. Uses config if None
    fmax : float, optional
        Maximum frequency. Uses config if None
        
    Returns:
    --------
    np.ndarray
        Baseline connectivity matrix (n_channels x n_channels)
    """
    print(f"      📊 Computing baseline connectivity...")
    
    # Use the same connectivity computation as trials
    baseline_connectivity = compute_trial_connectivity(
        baseline_data, srate, method, fmin, fmax
    )
    
    return baseline_connectivity

def apply_subtraction_baseline_normalization(trial_connectivity: np.ndarray, 
                                           baseline_connectivity: np.ndarray) -> np.ndarray:
    """
    Apply subtraction baseline normalization to connectivity matrix.
    
    Subtraction baseline normalization removes the baseline connectivity pattern
    from trial connectivity, highlighting task-related changes. This method is
    preferred for connectivity measures as it preserves the original scale and
    is more interpretable than other normalization methods.
    
    Formula: normalized_connectivity = trial_connectivity - baseline_connectivity
    
    Parameters:
    -----------
    trial_connectivity : np.ndarray
        Trial connectivity matrix
    baseline_connectivity : np.ndarray
        Baseline connectivity matrix
        
    Returns:
    --------
    np.ndarray
        Baseline-normalized connectivity matrix
    """
    return trial_connectivity - baseline_connectivity

def extract_baseline_period(data: np.ndarray, t_on: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Extract pre-experiment baseline period from continuous data.
    
    The pre-experiment baseline is extracted from the beginning of the recording
    until the first stimulus onset. This period represents the resting state
    without any task-related activity.
    
    Parameters:
    -----------
    data : np.ndarray
        Continuous signal data (n_channels x n_samples)
    t_on : np.ndarray
        Event onset times in samples
        
    Returns:
    --------
    tuple
        Baseline data and duration in seconds
    """
    baseline_start = 0
    baseline_end = t_on[0]  # Until first stimulus onset
    
    if baseline_end <= baseline_start:
        raise ValueError("Invalid baseline period: end time <= start time")
    
    baseline_data = data[:, baseline_start:baseline_end]
    baseline_duration_samples = baseline_end - baseline_start
    
    return baseline_data, baseline_duration_samples

def get_connectivity_info() -> dict:
    """
    Get connectivity analysis configuration information.
    
    Returns:
    --------
    dict
        Dictionary containing connectivity parameters
    """
    return {
        'method': config.CONNECTIVITY_PARAMS['method'],
        'frequency_range': (config.CONNECTIVITY_PARAMS['fmin'], config.CONNECTIVITY_PARAMS['fmax']),
        'normalization': config.CONNECTIVITY_PARAMS['baseline_normalization'],
        'description': f"Connectivity analysis using {config.CONNECTIVITY_PARAMS['method']} "
                      f"in {config.CONNECTIVITY_PARAMS['fmin']}-{config.CONNECTIVITY_PARAMS['fmax']} Hz "
                      f"with {config.CONNECTIVITY_PARAMS['baseline_normalization']} normalization"
    }