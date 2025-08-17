"""
3D Connectome Visualization for Motor Imagery ECoG Data

This module provides functions for creating 3D interactive connectome plots
to visualize functional connectivity patterns between brain regions during
motor imagery tasks (hand vs tongue).

Dependencies:
- nilearn
- numpy
- matplotlib (optional for saving static plots)

Author: [Negin Teimourpour]

"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from nilearn import plotting


def compute_roi_coordinates(
    electrode_locations: np.ndarray,
    brodmann_labels: np.ndarray,
    roi_list: List[int]
) -> np.ndarray:
    """
    Compute mean coordinates for each ROI based on electrode locations.
    
    Parameters:
    -----------
    electrode_locations : np.ndarray
        3D coordinates of electrodes (n_electrodes, 3)
    brodmann_labels : np.ndarray
        Brodmann area labels for each electrode
    roi_list : List[int]
        List of ROI numbers to compute coordinates for
        
    Returns:
    --------
    np.ndarray
        Mean coordinates for each ROI (n_rois, 3)
    """
    roi_coords = []
    
    for roi in roi_list:
        # Find electrodes belonging to this ROI
        roi_indices = np.where(brodmann_labels == roi)[0]
        
        if len(roi_indices) > 0:
            # Compute mean coordinate for this ROI
            mean_coord = electrode_locations[roi_indices].mean(axis=0)
        else:
            # If no electrodes for this ROI, use a random coordinate
            # (this shouldn't happen in practice with proper ROI selection)
            print(f"Warning: No electrodes found for ROI {roi}")
            mean_coord = np.random.rand(3) * 10
            
        roi_coords.append(mean_coord)
    
    return np.array(roi_coords)


def create_3d_connectome_plot(
    connectivity_matrix: np.ndarray,
    roi_coordinates: np.ndarray,
    roi_names: List[str] = None,
    edge_threshold: float = 0.01,
    edge_threshold_percentile: Optional[float] = None,
    node_color: str = 'black',
    node_size: int = 10,
    edge_cmap: str = 'plasma',
    title: str = '3D Connectome',
    display_mode: str = 'interactive'
) -> Any:
    """
    Create a 3D connectome visualization.
    
    Parameters:
    -----------
    connectivity_matrix : np.ndarray
        Symmetric connectivity matrix (n_rois, n_rois)
    roi_coordinates : np.ndarray
        3D coordinates of ROIs (n_rois, 3)
    roi_names : List[str], optional
        Names of ROIs for labeling
    edge_threshold : float, default=0.01
        Minimum connection strength to display
    edge_threshold_percentile : float, optional
        If provided, use this percentile as threshold instead
    node_color : str, default='black'
        Color of nodes
    node_size : int, default=10
        Size of nodes
    edge_cmap : str, default='plasma'
        Colormap for edges
    title : str, default='3D Connectome'
        Plot title
    display_mode : str, default='interactive'
        'interactive' for HTML widget, 'static' for matplotlib figure
        
    Returns:
    --------
    nilearn view object or matplotlib figure
    """
    # Validate inputs
    n_rois = connectivity_matrix.shape[0]
    if connectivity_matrix.shape != (n_rois, n_rois):
        raise ValueError("Connectivity matrix must be square")
    
    if roi_coordinates.shape[0] != n_rois:
        raise ValueError("Number of ROI coordinates must match connectivity matrix size")
    
    # Set threshold
    if edge_threshold_percentile is not None:
        # Use upper triangle to avoid double-counting connections
        upper_triangle = connectivity_matrix[np.triu_indices(n_rois, k=1)]
        threshold = np.percentile(upper_triangle, edge_threshold_percentile)
        print(f"Using {edge_threshold_percentile}th percentile threshold: {threshold:.4f}")
    else:
        threshold = edge_threshold
        print(f"Using fixed threshold: {threshold:.4f}")
    
    # Count connections above threshold
    n_connections = np.sum(connectivity_matrix > threshold)
    print(f"Displaying {n_connections} connections above threshold")
    
    # Create the plot
    if display_mode == 'interactive':
        view = plotting.view_connectome(
            adjacency_matrix=connectivity_matrix,
            node_coords=roi_coordinates,
            edge_threshold=threshold,
            node_color=node_color,
            node_size=node_size,
            edge_cmap=edge_cmap,
            title=title,
        )
        return view
    
    else:  # static plot
        fig = plotting.plot_connectome(
            adjacency_matrix=connectivity_matrix,
            node_coords=roi_coordinates,
            edge_threshold=threshold,
            node_color=node_color,
            node_size=node_size,
            edge_cmap=edge_cmap,
            title=title,
            display_mode='lzry',  # show multiple views
            colorbar=True
        )
        return fig


def create_connectivity_comparison_plots(
    hand_connectivity: np.ndarray,
    tongue_connectivity: np.ndarray,
    roi_coordinates: np.ndarray,
    roi_names: List[str] = None,
    edge_threshold_percentile: float = 90,
    save_static: bool = False,
    output_dir: str = "plots"
) -> Dict[str, Any]:
    """
    Create comparison plots for hand vs tongue connectivity.
    
    Parameters:
    -----------
    hand_connectivity : np.ndarray
        Connectivity matrix for hand imagery
    tongue_connectivity : np.ndarray
        Connectivity matrix for tongue imagery
    roi_coordinates : np.ndarray
        3D coordinates of ROIs
    roi_names : List[str], optional
        Names of ROIs
    edge_threshold_percentile : float, default=90
        Percentile threshold for edge display
    save_static : bool, default=False
        Whether to save static matplotlib versions
    output_dir : str, default="plots"
        Directory to save plots if save_static=True
        
    Returns:
    --------
    Dict containing the plot objects
    """
    plots = {}
    
    # Hand imagery connectome
    print("\n=== Creating Hand Imagery Connectome ===")
    hand_view = create_3d_connectome_plot(
        connectivity_matrix=hand_connectivity,
        roi_coordinates=roi_coordinates,
        roi_names=roi_names,
        edge_threshold_percentile=edge_threshold_percentile,
        title='3D Connectome - Hand Imagery',
        edge_cmap='Reds'
    )
    plots['hand'] = hand_view
    
    # Tongue imagery connectome  
    print("\n=== Creating Tongue Imagery Connectome ===")
    tongue_view = create_3d_connectome_plot(
        connectivity_matrix=tongue_connectivity,
        roi_coordinates=roi_coordinates,
        roi_names=roi_names,
        edge_threshold_percentile=edge_threshold_percentile,
        title='3D Connectome - Tongue Imagery',
        edge_cmap='Blues'
    )
    plots['tongue'] = tongue_view
    
    # Difference connectome (Hand - Tongue)
    print("\n=== Creating Difference Connectome ===")
    diff_connectivity = hand_connectivity - tongue_connectivity
    diff_view = create_3d_connectome_plot(
        connectivity_matrix=np.abs(diff_connectivity),  # Use absolute differences
        roi_coordinates=roi_coordinates,
        roi_names=roi_names,
        edge_threshold_percentile=edge_threshold_percentile,
        title='3D Connectome - |Hand - Tongue| Differences',
        edge_cmap='plasma'
    )
    plots['difference'] = diff_view
    
    # Save static versions if requested
    if save_static:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n=== Saving Static Plots to {output_dir} ===")
        
        # Static hand plot
        hand_static = create_3d_connectome_plot(
            connectivity_matrix=hand_connectivity,
            roi_coordinates=roi_coordinates,
            roi_names=roi_names,
            edge_threshold_percentile=edge_threshold_percentile,
            title='Hand Imagery Connectivity',
            display_mode='static',
            edge_cmap='Reds'
        )
        plt.savefig(f"{output_dir}/hand_connectome.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Static tongue plot
        tongue_static = create_3d_connectome_plot(
            connectivity_matrix=tongue_connectivity,
            roi_coordinates=roi_coordinates,
            roi_names=roi_names,
            edge_threshold_percentile=edge_threshold_percentile,
            title='Tongue Imagery Connectivity',
            display_mode='static',
            edge_cmap='Blues'
        )
        plt.savefig(f"{output_dir}/tongue_connectome.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Static plots saved to {output_dir}/")
    
    return plots


def analyze_connectivity_differences(
    hand_connectivity: np.ndarray,
    tongue_connectivity: np.ndarray,
    roi_names: List[str] = None,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Analyze the strongest connectivity differences between conditions.
    
    Parameters:
    -----------
    hand_connectivity : np.ndarray
        Hand imagery connectivity matrix
    tongue_connectivity : np.ndarray
        Tongue imagery connectivity matrix
    roi_names : List[str], optional
        ROI names for labeling
    top_k : int, default=10
        Number of top connections to report
        
    Returns:
    --------
    Dict containing analysis results
    """
    # Compute difference matrix
    diff_matrix = hand_connectivity - tongue_connectivity
    
    # Get upper triangle indices (avoid double counting)
    n_rois = diff_matrix.shape[0]
    triu_indices = np.triu_indices(n_rois, k=1)
    
    # Extract upper triangle values
    diff_values = diff_matrix[triu_indices]
    
    # Find strongest positive and negative differences
    pos_indices = np.argsort(diff_values)[-top_k:][::-1]  # Top positive (hand > tongue)
    neg_indices = np.argsort(diff_values)[:top_k]         # Top negative (tongue > hand)
    
    results = {
        'diff_matrix': diff_matrix,
        'mean_diff': np.mean(np.abs(diff_values)),
        'std_diff': np.std(diff_values),
        'max_pos_diff': np.max(diff_values),
        'max_neg_diff': np.min(diff_values)
    }
    
    # Report top connections
    if roi_names is not None:
        print(f"\n=== Top {top_k} Connections: Hand > Tongue ===")
        for i, idx in enumerate(pos_indices):
            row, col = triu_indices[0][idx], triu_indices[1][idx]
            value = diff_values[idx]
            print(f"{i+1:2d}. {roi_names[row]} - {roi_names[col]}: {value:+.4f}")
        
        print(f"\n=== Top {top_k} Connections: Tongue > Hand ===")
        for i, idx in enumerate(neg_indices):
            row, col = triu_indices[0][idx], triu_indices[1][idx]
            value = diff_values[idx]
            print(f"{i+1:2d}. {roi_names[row]} - {roi_names[col]}: {value:+.4f}")
    
    results['top_hand_connections'] = pos_indices
    results['top_tongue_connections'] = neg_indices
    
    return results


def create_circle_connectivity_plot(
    connectivity_matrix: np.ndarray,
    roi_names: List[str],
    n_lines: int = 20,
    title: str = 'ROI Connectivity',
    colormap: str = 'plasma',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> None:
    """
    Create a circular connectivity plot using MNE-connectivity.
    
    Parameters:
    -----------
    connectivity_matrix : np.ndarray
        Connectivity matrix to plot
    roi_names : List[str]
        Names of ROIs
    n_lines : int, default=20
        Number of strongest connections to show
    title : str, default='ROI Connectivity'
        Plot title
    colormap : str, default='plasma'
        Colormap for connections
    vmin, vmax : float, optional
        Color scale limits
    """
    try:
        from mne_connectivity.viz import plot_connectivity_circle
        
        # Set color scale limits if not provided
        if vmin is None or vmax is None:
            abs_max = np.max(np.abs(connectivity_matrix))
            vmin = -abs_max
            vmax = abs_max
        
        plot_connectivity_circle(
            connectivity_matrix, 
            roi_names,
            n_lines=n_lines,
            title=title,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            show=True, 
            facecolor='white', 
            textcolor='black'
        )
        
    except ImportError:
        print("Warning: mne-connectivity not available. Skipping circle plot.")
        print("Install with: pip install mne-connectivity")


# Example usage function
def example_usage():
    """
    Example of how to use the visualization functions.
    This would typically be called from your main analysis script.
    """
    # This is pseudo-code showing how to integrate with your main script
    
    # Assuming you have these variables from your main analysis:
    # - roi_matrix_hand: connectivity matrix for hand trials
    # - roi_matrix_tongue: connectivity matrix for tongue trials  
    # - electrode_locations_valid: electrode coordinates
    # - brodmann_labels_valid: electrode ROI labels
    # - common_rois: list of ROI numbers
    
    # Example data (replace with your actual data)
    # roi_matrix_hand = ...
    # roi_matrix_tongue = ...
    # electrode_locations_valid = ...
    # brodmann_labels_valid = ...
    # common_rois = [4, 6, 8, 9, 44, 45, 46, 47]
    
    # Compute ROI coordinates
    # roi_coords = compute_roi_coordinates(
    #     electrode_locations_valid, 
    #     brodmann_labels_valid, 
    #     common_rois
    # )
    
    # Create ROI names
    # roi_names = [f'BA{roi}' for roi in common_rois]
    
    # Create comparison plots
    # plots = create_connectivity_comparison_plots(
    #     hand_connectivity=roi_matrix_hand,
    #     tongue_connectivity=roi_matrix_tongue,
    #     roi_coordinates=roi_coords,
    #     roi_names=roi_names,
    #     edge_threshold_percentile=90,
    #     save_static=True
    # )
    
    # Analyze differences
    # diff_analysis = analyze_connectivity_differences(
    #     roi_matrix_hand, 
    #     roi_matrix_tongue, 
    #     roi_names, 
    #     top_k=10
    # )
    
    # Create circle plot for differences
    # create_circle_connectivity_plot(
    #     diff_analysis['diff_matrix'],
    #     roi_names,
    #     n_lines=20,
    #     title='Connectivity Differences (Hand - Tongue)',
    #     colormap='RdBu_r'
    # )
    
    pass


def add_visualization_to_main_analysis(
    all_roi_matrices: np.ndarray,
    all_labels: np.ndarray,
    electrode_locations_valid: np.ndarray,
    brodmann_labels_valid: np.ndarray,
    common_rois: list,
    save_plots: bool = True
):
    """
    Add 3D connectome visualization to the main analysis pipeline.
    
    Parameters:
    -----------
    all_roi_matrices : np.ndarray
        All ROI connectivity matrices (n_trials, n_rois, n_rois)
    all_labels : np.ndarray  
        Trial labels (0=tongue, 1=hand)
    electrode_locations_valid : np.ndarray
        Valid electrode coordinates
    brodmann_labels_valid : np.ndarray
        Valid electrode ROI labels
    common_rois : list
        List of ROI numbers used in analysis
    save_plots : bool, default=True
        Whether to save static versions of plots
    """
    
    print("\n" + "="*60)
    print("    3D CONNECTOME VISUALIZATION")
    print("="*60)
    
    # Separate trials by condition
    hand_trials = all_roi_matrices[all_labels == 1]  # Hand = 1
    tongue_trials = all_roi_matrices[all_labels == 0]  # Tongue = 0
    
    print(f"Hand trials: {len(hand_trials)}")
    print(f"Tongue trials: {len(tongue_trials)}")
    
    # Compute average connectivity matrices
    print("\n📊 Computing average connectivity matrices...")
    roi_matrix_hand = np.mean(hand_trials, axis=0)
    roi_matrix_tongue = np.mean(tongue_trials, axis=0)
    
    print(f"   • Hand connectivity shape: {roi_matrix_hand.shape}")
    print(f"   • Tongue connectivity shape: {roi_matrix_tongue.shape}")
    
    # Compute ROI coordinates from electrode locations
    print("\n📍 Computing ROI coordinates...")
    roi_coords = compute_roi_coordinates(
        electrode_locations_valid, 
        brodmann_labels_valid, 
        common_rois
    )
    print(f"   • ROI coordinates shape: {roi_coords.shape}")
    
    # Create ROI names for labeling
    roi_names = [f'BA{roi}' for roi in common_rois]
    print(f"   • ROIs: {roi_names}")
    
    # Create 3D connectome visualizations
    print("\n🎨 Creating 3D connectome visualizations...")
    plots = create_connectivity_comparison_plots(
        hand_connectivity=roi_matrix_hand,
        tongue_connectivity=roi_matrix_tongue,
        roi_coordinates=roi_coords,
        roi_names=roi_names,
        edge_threshold_percentile=90,
        save_static=save_plots,
        output_dir="connectome_plots"
    )
    
    # Analyze connectivity differences
    print("\n🔍 Analyzing connectivity differences...")
    diff_analysis = analyze_connectivity_differences(
        roi_matrix_hand, 
        roi_matrix_tongue, 
        roi_names, 
        top_k=10
    )
    
    print(f"\nConnectivity Difference Statistics:")
    print(f"   • Mean absolute difference: {diff_analysis['mean_diff']:.4f}")
    print(f"   • Std of differences: {diff_analysis['std_diff']:.4f}")
    print(f"   • Max positive difference: {diff_analysis['max_pos_diff']:+.4f}")
    print(f"   • Max negative difference: {diff_analysis['max_neg_diff']:+.4f}")
    
    # Create circular connectivity plot for differences
    print("\n⭕ Creating circular connectivity plot...")
    try:
        create_circle_connectivity_plot(
            diff_analysis['diff_matrix'],
            roi_names,
            n_lines=20,
            title='Connectivity Differences (Hand - Tongue)',
            colormap='RdBu_r'
        )
        print("   ✅ Circle plot created successfully")
    except Exception as e:
        print(f"   ⚠️ Could not create circle plot: {e}")
    
    print("\n🎉 3D visualization completed!")
    
    # Return useful results
    return {
        'plots': plots,
        'roi_matrix_hand': roi_matrix_hand,
        'roi_matrix_tongue': roi_matrix_tongue,
        'roi_coords': roi_coords,
        'roi_names': roi_names,
        'diff_analysis': diff_analysis
    }


if __name__ == "__main__":
    print("3D Connectome Visualization Module")
    print("Import this module and use the functions in your main analysis script.")
    print("See example_usage() function for integration examples.")