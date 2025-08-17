import numpy as np
from scipy.signal import butter, filtfilt
from typing import Tuple
import config

def design_bandpass_filter(lowcut: float, highcut: float, fs: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a Butterworth bandpass filter.
    
    Parameters:
    -----------
    lowcut : float
        Lower cutoff frequency in Hz
    highcut : float
        Upper cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    order : int
        Filter order
        
    Returns:
    --------
    tuple
        Filter coefficients (b, a)
    """
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    
    # Design filter coefficients
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data: np.ndarray, lowcut: float = None, highcut: float = None, 
                         fs: float = None, order: int = None) -> np.ndarray:
    """
    Apply bandpass filter to neural data.
    
    This function filters the data to retain frequencies within the specified range.
    Commonly used to isolate specific frequency bands (e.g., gamma: 70-100 Hz).
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal data (channels x time)
    lowcut : float, optional
        Lower cutoff frequency. Uses config if None
    highcut : float, optional
        Upper cutoff frequency. Uses config if None
    fs : float
        Sampling frequency
    order : int, optional
        Filter order. Uses config if None
        
    Returns:
    --------
    np.ndarray
        Filtered signal data
    """
    # Use config defaults if parameters not provided
    if lowcut is None:
        lowcut = config.FILTER_PARAMS['lowcut']
    if highcut is None:
        highcut = config.FILTER_PARAMS['highcut']
    if order is None:
        order = config.FILTER_PARAMS['order']
    
    # Design and apply filter
    b, a = design_bandpass_filter(lowcut, highcut, fs, order)
    
    # Apply filter along time axis (last dimension)
    filtered_data = filtfilt(b, a, data, axis=-1)
    
    return filtered_data

def get_filter_info(lowcut: float = None, highcut: float = None) -> str:
    """
    Get human-readable filter information.
    
    Parameters:
    -----------
    lowcut : float, optional
        Lower cutoff frequency
    highcut : float, optional  
        Upper cutoff frequency
        
    Returns:
    --------
    str
        Filter description
    """
    if lowcut is None:
        lowcut = config.FILTER_PARAMS['lowcut']
    if highcut is None:
        highcut = config.FILTER_PARAMS['highcut']
    
    return f"Bandpass filter: {lowcut}-{highcut} Hz"