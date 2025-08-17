import numpy as np
from typing import Tuple, List, Dict, Any
import config

def load_motor_imagery_data(data_path: str = None) -> np.ndarray:
    """
    Load motor imagery dataset.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the dataset file. If None, uses config.DATA_PATH
        
    Returns:
    --------
    np.ndarray
        Array containing all subject data
    """
    if data_path is None:
        data_path = config.DATA_PATH
    
    try:
        alldat = np.load(data_path, allow_pickle=True)['dat']
        print(f"✅ Successfully loaded data: {len(alldat)} subjects detected")
        return alldat
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def extract_subject_data(sub_data: Dict[str, Any]) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and preprocess data from a single subject.
    
    Parameters:
    -----------
    sub_data : dict
        Subject data dictionary
        
    Returns:
    --------
    tuple
        V_scaled (signals), srate (sampling rate), t_on (event times), 
        stim_id (stimulus IDs), brodmann_labels (brain regions)
    """
    # Extract raw data and scale to microvolts
    V = sub_data['V'] * sub_data['scale_uv']
    V = V.T  # Transpose to (channels, time)
    
    # Extract other parameters
    srate = sub_data['srate']
    t_on = sub_data['t_on']
    stim_id = sub_data['stim_id']
    brodmann_labels = sub_data['Brodmann_Area']
    
    return V, srate, t_on, stim_id, brodmann_labels

def get_valid_channels(V: np.ndarray, brodmann_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Identify and return valid channels (non-NaN, non-zero).
    
    Parameters:
    -----------
    V : np.ndarray
        Signal data (channels x time)
    brodmann_labels : np.ndarray
        Brodmann area labels for each channel
        
    Returns:
    --------
    tuple
        Valid signals, valid labels, valid channel mask
    """
    # Create mask for valid channels (not all NaN or all zero)
    mask_valid = np.logical_and(
        ~np.isnan(V).all(axis=1), 
        ~(V == 0).all(axis=1)
    )
    
    V_valid = V[mask_valid, :]
    brodmann_labels_valid = np.array(brodmann_labels)[mask_valid]
    
    print(f"      📊 Valid channels: {np.sum(mask_valid)}/{len(mask_valid)}")
    
    return V_valid, brodmann_labels_valid, mask_valid

def extract_epochs(data: np.ndarray, t_on: np.ndarray, stim_id: np.ndarray, 
                   target_stim: int, srate: float, epoch_ms: int = 3000) -> np.ndarray:
    """
    Extract epochs for a specific stimulus type.
    
    Parameters:
    -----------
    data : np.ndarray
        Signal data (channels x time)
    t_on : np.ndarray
        Event onset times
    stim_id : np.ndarray
        Stimulus identifiers
    target_stim : int
        Target stimulus ID to extract
    srate : float
        Sampling rate
    epoch_ms : int
        Epoch duration in milliseconds
        
    Returns:
    --------
    np.ndarray
        Extracted epochs (n_trials x n_channels x n_samples)
    """
    # Find indices for target stimulus
    idx = np.where(stim_id == target_stim)[0]
    epoch_samples = int(epoch_ms * srate / 1000)
    
    epochs = []
    for i in idx:
        start = t_on[i]
        stop = start + epoch_samples
        
        # Check if epoch fits within data
        if stop <= data.shape[1]:
            epochs.append(data[:, start:stop])
    
    return np.array(epochs) if epochs else np.array([]).reshape(0, data.shape[0], epoch_samples)

def save_results(results: Dict[str, Any], filename: str = "analysis_results.npz") -> None:
    """
    Save analysis results to file.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing analysis results
    filename : str
        Output filename
    """
    try:
        np.savez_compressed(filename, **results)
        print(f"💾 Results saved to: {filename}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def print_data_summary(all_labels: np.ndarray) -> None:
    """
    Print summary statistics of the processed data.
    
    Parameters:
    -----------
    all_labels : np.ndarray
        Array of class labels
    """
    total_trials = len(all_labels)
    hand_trials = np.sum(all_labels == 1)
    tongue_trials = np.sum(all_labels == 0)
    
    print(f"\n📊 Data Summary:")
    print(f"   • Total trials: {total_trials}")
    print(f"   • Hand trials: {hand_trials} ({hand_trials/total_trials*100:.1f}%)")
    print(f"   • Tongue trials: {tongue_trials} ({tongue_trials/total_trials*100:.1f}%)")