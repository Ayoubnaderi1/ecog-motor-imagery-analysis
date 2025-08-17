import numpy as np
from typing import List
import config

def convert_connectivity_to_roi_matrix(connectivity_matrix: np.ndarray, 
                                      brodmann_labels: np.ndarray, 
                                      common_rois: List[str] = None) -> np.ndarray:
    """
    Convert electrode-level connectivity matrix to ROI-level matrix.
    
    This function aggregates connectivity values between electrodes belonging
    to the same brain regions (Brodmann areas). Multiple electrodes in the same
    ROI are averaged to create a single representative value.
    
    Parameters:
    -----------
    connectivity_matrix : np.ndarray
        Electrode-level connectivity matrix (n_electrodes x n_electrodes)
    brodmann_labels : np.ndarray
        Brodmann area labels for each electrode
    common_rois : list, optional
        List of ROI names. Uses config if None
        
    Returns:
    --------
    np.ndarray
        ROI-level connectivity matrix (n_rois x n_rois)
    """
    if common_rois is None:
        common_rois = config.COMMON_ROIS
    
    n_rois = len(common_rois)
    roi_matrix = np.zeros((n_rois, n_rois))
    
    # For each pair of ROIs
    for i, roi1 in enumerate(common_rois):
        for j, roi2 in enumerate(common_rois):
            # Find electrodes belonging to each ROI
            idx_i = np.where(brodmann_labels == roi1)[0]
            idx_j = np.where(brodmann_labels == roi2)[0]
            
            # If both ROIs have electrodes, compute average connectivity
            if len(idx_i) > 0 and len(idx_j) > 0:
                # Extract submatrix for this ROI pair
                submatrix = connectivity_matrix[np.ix_(idx_i, idx_j)]
                # Average all connections between these ROIs
                roi_matrix[i, j] = np.mean(submatrix)
    
    return roi_matrix

def get_roi_names(short_names: bool = True) -> List[str]:
    """
    Get ROI names for labeling.
    
    Parameters:
    -----------
    short_names : bool
        If True, return abbreviated names (e.g., 'BA4'). 
        If False, return full names (e.g., 'Brodmann area 4')
        
    Returns:
    --------
    list
        List of ROI names
    """
    if short_names:
        # Convert "Brodmann area X" to "BAX"
        return [roi.replace('Brodmann area ', 'BA') for roi in config.COMMON_ROIS]
    else:
        return config.COMMON_ROIS.copy()

def validate_roi_coverage(brodmann_labels: np.ndarray, common_rois: List[str] = None) -> dict:
    """
    Validate ROI coverage in the dataset.
    
    Check which ROIs have electrodes and how many electrodes per ROI.
    
    Parameters:
    -----------
    brodmann_labels : np.ndarray
        Brodmann area labels for each electrode
    common_rois : list, optional
        List of ROI names. Uses config if None
        
    Returns:
    --------
    dict
        Dictionary with ROI coverage information
    """
    if common_rois is None:
        common_rois = config.COMMON_ROIS
    
    roi_coverage = {}
    
    for roi in common_rois:
        n_electrodes = np.sum(brodmann_labels == roi)
        roi_coverage[roi] = n_electrodes
    
    # Summary statistics
    covered_rois = sum(1 for count in roi_coverage.values() if count > 0)
    total_electrodes = sum(roi_coverage.values())
    
    coverage_info = {
        'roi_counts': roi_coverage,
        'covered_rois': covered_rois,
        'total_rois': len(common_rois),
        'coverage_percentage': (covered_rois / len(common_rois)) * 100,
        'total_electrodes': total_electrodes
    }
    
    return coverage_info

def print_roi_coverage(coverage_info: dict) -> None:
    """
    Print ROI coverage information in a formatted way.
    
    Parameters:
    -----------
    coverage_info : dict
        Output from validate_roi_coverage()
    """
    print(f"      🧠 ROI Coverage:")
    print(f"         • Covered ROIs: {coverage_info['covered_rois']}/{coverage_info['total_rois']} "
          f"({coverage_info['coverage_percentage']:.1f}%)")
    print(f"         • Total electrodes: {coverage_info['total_electrodes']}")
    
    # Print electrode count per ROI
    for roi, count in coverage_info['roi_counts'].items():
        if count > 0:
            roi_short = roi.replace('Brodmann area ', 'BA')
            print(f"         • {roi_short}: {count} electrodes")

def get_roi_pair_name(i: int, j: int, short_names: bool = True) -> str:
    """
    Get name for a specific ROI pair connection.
    
    Parameters:
    -----------
    i : int
        First ROI index
    j : int
        Second ROI index  
    short_names : bool
        Use abbreviated names
        
    Returns:
    --------
    str
        ROI pair name (e.g., "BA4-BA6")
    """
    roi_names = get_roi_names(short_names)
    return f"{roi_names[i]}-{roi_names[j]}"