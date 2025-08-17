"""
Motor Imagery ECoG Analysis - Main Execution Script (Fixed for Visualization)

This script performs connectivity-based classification of motor imagery tasks
using ECoG data with 3D connectome visualization support.
"""
from visualization_3d import add_visualization_to_main_analysis
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mne_connectivity import spectral_connectivity_epochs
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from scipy.signal import butter, filtfilt

import numpy as np
import sys
from pathlib import Path

# Add utils and models to path
sys.path.append(str(Path(__file__).parent))

# Import modules
import config
from utils.io_utils import (
    load_motor_imagery_data, extract_subject_data, get_valid_channels,
    extract_epochs, print_data_summary, save_results
)
from utils.filtering import apply_bandpass_filter, get_filter_info
from utils.connectivity import (
    compute_trial_connectivity, compute_baseline_connectivity,
    apply_subtraction_baseline_normalization, extract_baseline_period,
    get_connectivity_info
)
from utils.roi import (
    convert_connectivity_to_roi_matrix, validate_roi_coverage, 
    print_roi_coverage
)
from utils.feature_extraction import prepare_features_for_classification
from models.classifier import run_complete_classification_pipeline

def process_single_subject(sub_idx: int, sub_data: dict) -> tuple:
    """
    Process a single subject's data through the complete pipeline.
    
    Parameters:
    -----------
    sub_idx : int
        Subject index
    sub_data : dict
        Subject data dictionary
        
    Returns:
    --------
    tuple
        (roi_matrices_list, labels_list, subject_info) or (None, None, None) if processing failed
        subject_info contains: brodmann_labels_valid, electrode_locations_valid (if available)
    """
    print(f"\n⏳ Processing subject {sub_idx}...")
    
    try:
        # Extract subject data
        V, srate, t_on, stim_id, brodmann_labels = extract_subject_data(sub_data)
        
        # Try to extract electrode locations if available
        electrode_locations = None
        if 'locs' in sub_data:
            electrode_locations = sub_data['locs']
       
        
        # Apply bandpass filter
        print(f"   🔧 Applying filter: {get_filter_info()}")
        V_filtered = apply_bandpass_filter(V, fs=srate)
        
        # Get valid channels
        V_valid, brodmann_labels_valid, mask_valid = get_valid_channels(
            V_filtered, brodmann_labels
        )
        
        # Apply same mask to electrode locations if available
        electrode_locations_valid = None
        if electrode_locations is not None:
            electrode_locations_valid = electrode_locations[mask_valid]
            print(f"   📍 Electrode locations: {electrode_locations_valid.shape}")
        
        # Validate ROI coverage
        coverage_info = validate_roi_coverage(brodmann_labels_valid)
        print_roi_coverage(coverage_info)
        
        if coverage_info['covered_rois'] < 3:  # Need at least 3 ROIs
            print(f"   ⚠️ Insufficient ROI coverage, skipping subject {sub_idx}")
            return None, None, None
        
        # Extract pre-experiment baseline
        print(f"   📈 Extracting pre-experiment baseline...")
        baseline_data, baseline_duration = extract_baseline_period(V_valid, t_on)
        baseline_duration_sec = baseline_duration / srate
        print(f"      ⏱️ Baseline duration: {baseline_duration_sec:.2f} seconds")
        
        if baseline_duration_sec < 1.0:  # Need at least 1 second
            print(f"   ⚠️ Insufficient baseline duration, skipping subject {sub_idx}")
            return None, None, None
        
        # Compute baseline connectivity
        baseline_connectivity = compute_baseline_connectivity(baseline_data, srate)
        print(f"      ✅ Baseline connectivity computed successfully")
        
        # Extract epochs for both tasks
        hand_trials = extract_epochs(
            V_valid, t_on, stim_id, config.TASK_PARAMS['hand_stim_id'], srate
        )
        tongue_trials = extract_epochs(
            V_valid, t_on, stim_id, config.TASK_PARAMS['tongue_stim_id'], srate
        )
        
        print(f"   • Hand trials: {len(hand_trials)}, Tongue trials: {len(tongue_trials)}")
        
        if len(hand_trials) == 0 or len(tongue_trials) == 0:
            print(f"   ⚠️ No trials found for one or both conditions, skipping subject {sub_idx}")
            return None, None, None
        
        # Process trials
        roi_matrices = []
        labels = []
        
        # Process hand trials
        print(f"   🔄 Processing hand trials...")
        for trial_idx, single_trial in enumerate(hand_trials):
            # Compute trial connectivity
            trial_connectivity = compute_trial_connectivity(single_trial, srate)
            
            # Apply subtraction baseline normalization
            normalized_connectivity = apply_subtraction_baseline_normalization(
                trial_connectivity, baseline_connectivity
            )
            
            # Convert to ROI matrix
            roi_matrix = convert_connectivity_to_roi_matrix(
                normalized_connectivity, brodmann_labels_valid
            )
            
            roi_matrices.append(roi_matrix)
            labels.append(1)  # Hand = 1
        
        # Process tongue trials  
        print(f"   🔄 Processing tongue trials...")
        for trial_idx, single_trial in enumerate(tongue_trials):
            # Compute trial connectivity
            trial_connectivity = compute_trial_connectivity(single_trial, srate)
            
            # Apply subtraction baseline normalization
            normalized_connectivity = apply_subtraction_baseline_normalization(
                trial_connectivity, baseline_connectivity
            )
            
            # Convert to ROI matrix
            roi_matrix = convert_connectivity_to_roi_matrix(
                normalized_connectivity, brodmann_labels_valid
            )
            
            roi_matrices.append(roi_matrix)
            labels.append(0)  # Tongue = 0
        
        print(f"   ✅ Subject {sub_idx} processed successfully")
        print(f"      📊 Total trials: {len(roi_matrices)} (Hand: {len(hand_trials)}, Tongue: {len(tongue_trials)})")
        
        # Package subject info for visualization
        subject_info = {
            'brodmann_labels_valid': brodmann_labels_valid,
            'electrode_locations_valid': electrode_locations_valid,
            'subject_idx': sub_idx
        }
        
        return roi_matrices, labels, subject_info
        
    except Exception as e:
        print(f"   ❌ Error processing subject {sub_idx}: {e}")
        return None, None, None


def create_comprehensive_visualization_report(
    all_roi_matrices: np.ndarray,
    all_labels: np.ndarray,
    electrode_locations_valid: np.ndarray,
    brodmann_labels_valid: np.ndarray,
    common_rois: list,
    save_plots: bool = True,
    output_dir: str = "connectome_plots"
) -> dict:
    """
    Create a comprehensive visualization report with multiple plot types.
    
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
    output_dir : str, default="connectome_plots"
        Directory to save plots
        
    Returns:
    --------
    dict
        Dictionary containing all visualization results
    """
    print(f"\n" + "="*60)
    print(f"    COMPREHENSIVE VISUALIZATION REPORT")
    print(f"="*60)
    
    try:
        # Import visualization functions
        from visualization_3d import (
            add_visualization_to_main_analysis,
            create_connectivity_comparison_plots,
            analyze_connectivity_differences,
            create_circle_connectivity_plot,
            compute_roi_coordinates
        )
        
        # Create main 3D visualizations
        print(f"\n🎨 Creating 3D Connectome Visualizations...")
        viz_results = add_visualization_to_main_analysis(
            all_roi_matrices=all_roi_matrices,
            all_labels=all_labels,
            electrode_locations_valid=electrode_locations_valid,
            brodmann_labels_valid=brodmann_labels_valid,
            common_rois=common_rois,
            save_plots=save_plots
        )
        
        # Create additional analysis plots
        print(f"\n📊 Creating Additional Analysis Plots...")
        
        # Compute ROI coordinates
        roi_coords = compute_roi_coordinates(
            electrode_locations_valid, 
            brodmann_labels_valid, 
            common_rois
        )
        roi_names = [f'BA{roi}' for roi in common_rois]
        
        # Separate trials by condition
        hand_trials = all_roi_matrices[all_labels == 1]
        tongue_trials = all_roi_matrices[all_labels == 0]
        
        # Compute average connectivity matrices
        roi_matrix_hand = np.mean(hand_trials, axis=0)
        roi_matrix_tongue = np.mean(tongue_trials, axis=0)
        
        # Create connectivity comparison plots
        comparison_plots = create_connectivity_comparison_plots(
            hand_connectivity=roi_matrix_hand,
            tongue_connectivity=roi_matrix_tongue,
            roi_coordinates=roi_coords,
            roi_names=roi_names,
            edge_threshold_percentile=90,
            save_static=save_plots,
            output_dir=output_dir
        )
        
        # Analyze connectivity differences
        diff_analysis = analyze_connectivity_differences(
            roi_matrix_hand, 
            roi_matrix_tongue, 
            roi_names, 
            top_k=15
        )
        
        # Try to create circle plot
        try:
            print(f"\n⭕ Creating Circular Connectivity Plot...")
            create_circle_connectivity_plot(
                diff_analysis['diff_matrix'],
                roi_names,
                n_lines=25,
                title='Connectivity Differences (Hand - Tongue)',
                colormap='RdBu_r'
            )
            circle_plot_created = True
        except Exception as e:
            print(f"   ⚠️ Circle plot creation failed: {e}")
            circle_plot_created = False
        
        # Create summary statistics
        print(f"\n📈 Creating Summary Statistics...")
        
        # Connectivity strength statistics
        hand_strength = np.mean(np.abs(roi_matrix_hand))
        tongue_strength = np.mean(np.abs(roi_matrix_tongue))
        
        # ROI-wise analysis
        roi_hand_strength = np.mean(np.abs(roi_matrix_hand), axis=1)
        roi_tongue_strength = np.mean(np.abs(roi_matrix_tongue), axis=1)
        
        summary_stats = {
            'overall_hand_strength': hand_strength,
            'overall_tongue_strength': tongue_strength,
            'strength_ratio': hand_strength / tongue_strength if tongue_strength > 0 else float('inf'),
            'roi_hand_strength': roi_hand_strength,
            'roi_tongue_strength': roi_tongue_strength,
            'n_hand_trials': len(hand_trials),
            'n_tongue_trials': len(tongue_trials),
            'circle_plot_created': circle_plot_created
        }
        
        # Print summary
        print(f"\n📋 Visualization Summary:")
        print(f"   • Hand trials: {summary_stats['n_hand_trials']}")
        print(f"   • Tongue trials: {summary_stats['n_tongue_trials']}")
        print(f"   • Overall hand connectivity: {hand_strength:.4f}")
        print(f"   • Overall tongue connectivity: {tongue_strength:.4f}")
        print(f"   • Hand/Tongue ratio: {summary_stats['strength_ratio']:.2f}")
        print(f"   • Mean connectivity difference: {diff_analysis['mean_diff']:.4f}")
        print(f"   • Circle plot: {'✅ Created' if circle_plot_created else '❌ Failed'}")
        
        # Combine all results
        comprehensive_results = {
            'main_visualization': viz_results,
            'comparison_plots': comparison_plots,
            'diff_analysis': diff_analysis,
            'summary_stats': summary_stats,
            'roi_coords': roi_coords,
            'roi_names': roi_names,
            'roi_matrix_hand': roi_matrix_hand,
            'roi_matrix_tongue': roi_matrix_tongue
        }
        
        print(f"\n✅ Comprehensive visualization report completed!")
        return comprehensive_results
        
    except Exception as e:
        print(f"❌ Error in comprehensive visualization: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_basic_connectivity_visualization(
    all_roi_matrices: np.ndarray,
    all_labels: np.ndarray,
    common_rois: list,
    save_plots: bool = True,
    output_dir: str = "connectome_plots"
) -> dict:
    """
    Create basic connectivity visualizations when electrode location data is not available.
    
    Parameters:
    -----------
    all_roi_matrices : np.ndarray
        All ROI connectivity matrices (n_trials, n_rois, n_rois)
    all_labels : np.ndarray
        Trial labels (0=tongue, 1=hand)
    common_rois : list
        List of ROI numbers used in analysis
    save_plots : bool, default=True
        Whether to save static versions of plots
    output_dir : str, default="connectome_plots"
        Directory to save plots
        
    Returns:
    --------
    dict
        Dictionary containing basic visualization results
    """
    print(f"\n" + "="*60)
    print(f"    BASIC CONNECTIVITY VISUALIZATION")
    print(f"="*60)
    
    try:
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create output directory
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
        
        # Separate trials by condition
        hand_trials = all_roi_matrices[all_labels == 1]
        tongue_trials = all_roi_matrices[all_labels == 0]
        
        print(f"   • Hand trials: {len(hand_trials)}")
        print(f"   • Tongue trials: {len(tongue_trials)}")
        
        # Compute average connectivity matrices
        roi_matrix_hand = np.mean(hand_trials, axis=0)
        roi_matrix_tongue = np.mean(tongue_trials, axis=0)
        
        # Create ROI names
        roi_names = [f'BA{roi}' for roi in common_rois]
        
        # Create connectivity heatmaps
        print(f"\n📊 Creating Connectivity Heatmaps...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Hand connectivity heatmap
        sns.heatmap(roi_matrix_hand, 
                   xticklabels=roi_names, 
                   yticklabels=roi_names,
                   cmap='Reds', 
                   ax=axes[0],
                   cbar_kws={'label': 'Connectivity Strength'})
        axes[0].set_title('Hand Imagery Connectivity')
        axes[0].set_xlabel('ROI')
        axes[0].set_ylabel('ROI')
        
        # Tongue connectivity heatmap
        sns.heatmap(roi_matrix_tongue, 
                   xticklabels=roi_names, 
                   yticklabels=roi_names,
                   cmap='Blues', 
                   ax=axes[1],
                   cbar_kws={'label': 'Connectivity Strength'})
        axes[1].set_title('Tongue Imagery Connectivity')
        axes[1].set_xlabel('ROI')
        axes[1].set_ylabel('ROI')
        
        # Difference heatmap
        diff_matrix = roi_matrix_hand - roi_matrix_tongue
        sns.heatmap(diff_matrix, 
                   xticklabels=roi_names, 
                   yticklabels=roi_names,
                   cmap='RdBu_r', 
                   center=0,
                   ax=axes[2],
                   cbar_kws={'label': 'Hand - Tongue'})
        axes[2].set_title('Connectivity Differences (Hand - Tongue)')
        axes[2].set_xlabel('ROI')
        axes[2].set_ylabel('ROI')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_dir}/connectivity_heatmaps.png", dpi=300, bbox_inches='tight')
            print(f"   ✅ Heatmaps saved to {output_dir}/connectivity_heatmaps.png")
        
        plt.show()
        
        # Create connectivity strength comparison
        print(f"\n📈 Creating Connectivity Strength Comparison...")
        
        # Compute overall connectivity strengths
        hand_strength = np.mean(np.abs(roi_matrix_hand))
        tongue_strength = np.mean(np.abs(roi_matrix_tongue))
        
        # ROI-wise connectivity strengths
        roi_hand_strength = np.mean(np.abs(roi_matrix_hand), axis=1)
        roi_tongue_strength = np.mean(np.abs(roi_matrix_tongue), axis=1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall comparison
        conditions = ['Hand', 'Tongue']
        strengths = [hand_strength, tongue_strength]
        colors = ['red', 'blue']
        
        bars = ax1.bar(conditions, strengths, color=colors, alpha=0.7)
        ax1.set_title('Overall Connectivity Strength')
        ax1.set_ylabel('Mean Connectivity Strength')
        ax1.set_ylim(0, max(strengths) * 1.1)
        
        # Add value labels on bars
        for bar, strength in zip(bars, strengths):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{strength:.4f}', ha='center', va='bottom')
        
        # ROI-wise comparison
        x = np.arange(len(roi_names))
        width = 0.35
        
        ax2.bar(x - width/2, roi_hand_strength, width, label='Hand', color='red', alpha=0.7)
        ax2.bar(x + width/2, roi_tongue_strength, width, label='Tongue', color='blue', alpha=0.7)
        
        ax2.set_title('ROI-wise Connectivity Strength')
        ax2.set_xlabel('ROI')
        ax2.set_ylabel('Mean Connectivity Strength')
        ax2.set_xticks(x)
        ax2.set_xticklabels(roi_names, rotation=45)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_dir}/connectivity_strength_comparison.png", dpi=300, bbox_inches='tight')
            print(f"   ✅ Strength comparison saved to {output_dir}/connectivity_strength_comparison.png")
        
        plt.show()
        
        # Create summary statistics
        summary_stats = {
            'overall_hand_strength': hand_strength,
            'overall_tongue_strength': tongue_strength,
            'strength_ratio': hand_strength / tongue_strength if tongue_strength > 0 else float('inf'),
            'roi_hand_strength': roi_hand_strength,
            'roi_tongue_strength': roi_tongue_strength,
            'n_hand_trials': len(hand_trials),
            'n_tongue_trials': len(tongue_trials),
            'mean_diff': np.mean(np.abs(diff_matrix)),
            'max_pos_diff': np.max(diff_matrix),
            'max_neg_diff': np.min(diff_matrix)
        }
        
        print(f"\n📋 Basic Visualization Summary:")
        print(f"   • Hand trials: {summary_stats['n_hand_trials']}")
        print(f"   • Tongue trials: {summary_stats['n_tongue_trials']}")
        print(f"   • Overall hand connectivity: {hand_strength:.4f}")
        print(f"   • Overall tongue connectivity: {tongue_strength:.4f}")
        print(f"   • Hand/Tongue ratio: {summary_stats['strength_ratio']:.2f}")
        print(f"   • Mean connectivity difference: {summary_stats['mean_diff']:.4f}")
        print(f"   • Max hand > tongue: {summary_stats['max_pos_diff']:+.4f}")
        print(f"   • Max tongue > hand: {summary_stats['max_neg_diff']:+.4f}")
        
        return {
            'roi_matrix_hand': roi_matrix_hand,
            'roi_matrix_tongue': roi_matrix_tongue,
            'diff_matrix': diff_matrix,
            'summary_stats': summary_stats,
            'roi_names': roi_names
        }
        
    except Exception as e:
        print(f"❌ Error in basic visualization: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main execution function.
    """
    print("="*60)
    print("    MOTOR IMAGERY ECoG CONNECTIVITY ANALYSIS")
    print("="*60)
    print(f"Configuration:")
    print(f"   • Data path: {config.DATA_PATH}")
    print(f"   • Filter: {get_filter_info()}")
    
    connectivity_info = get_connectivity_info()
    print(f"   • Connectivity: {connectivity_info['description']}")
    print(f"   • ROIs: {len(config.COMMON_ROIS)} brain regions")
    print(f"   • Features: {len(config.IMPORTANT_CONNECTIONS)} connections")
    
    # Load data
    print(f"\n📂 Loading dataset...")
    alldat = load_motor_imagery_data()
    n_subjects = len(alldat)
    
    # Process all subjects
    all_roi_matrices = []
    all_labels = []
    all_subject_info = []
    processed_subjects = 0
    
    for sub_idx in range(n_subjects):
        sub_data = alldat[sub_idx][0]  # Extract the actual data dict
        
        # Updated to handle 3 return values
        roi_matrices, labels, subject_info = process_single_subject(sub_idx, sub_data)
        
        if roi_matrices is not None and labels is not None:
            all_roi_matrices.extend(roi_matrices)
            all_labels.extend(labels)
            all_subject_info.append(subject_info)
            processed_subjects += 1
    
    # Convert to arrays
    all_roi_matrices = np.array(all_roi_matrices)
    all_labels = np.array(all_labels)
    
    print(f"\n📊 Processing Summary:")
    print(f"   • Subjects processed: {processed_subjects}/{n_subjects}")
    print(f"   • Normalization: {config.CONNECTIVITY_PARAMS['baseline_normalization']}")
    print_data_summary(all_labels)
    
    if len(all_roi_matrices) == 0:
        print(f"❌ No data to analyze. Exiting.")
        return
    
    # Feature extraction and validation
    print(f"\n🔧 Feature Extraction and Validation...")
    features, labels, feature_stats = prepare_features_for_classification(
        all_roi_matrices, all_labels
    )
    
    if not feature_stats['validation_passed']:
        print(f"❌ Feature validation failed. Please check data quality.")
        return
    
    # Run classification pipeline
    print(f"\n🤖 Starting Classification Pipeline...")
    results = run_complete_classification_pipeline(
        features, labels, feature_stats['feature_names']
    )
    
    # Prepare visualization data
    print(f"\n🎨 Preparing for 3D Visualization...")
    
    # Find a subject with electrode location data for visualization
    electrode_locations_for_viz = None
    brodmann_labels_for_viz = None
    
    for subject_info in all_subject_info:
        if subject_info['electrode_locations_valid'] is not None:
            electrode_locations_for_viz = subject_info['electrode_locations_valid']
            brodmann_labels_for_viz = subject_info['brodmann_labels_valid']
            print(f"   📍 Using electrode locations from subject {subject_info['subject_idx']}")
            break
    
    # Save results
    print(f"\n💾 Saving results...")
    save_dict = {
        'features': features,
        'labels': labels,
        'roi_matrices': all_roi_matrices,
        'feature_stats': feature_stats,
        'classification_results': results,
        'config': {
            'connectivity_params': config.CONNECTIVITY_PARAMS,
            'filter_params': config.FILTER_PARAMS,
            'roi_names': config.COMMON_ROIS,
            'important_connections': config.IMPORTANT_CONNECTIONS,
            'n_subjects_processed': processed_subjects
        }
    }
    
    # Add comprehensive 3D visualization if electrode data is available
    if electrode_locations_for_viz is not None and processed_subjects > 0:
        try:
            print(f"\n🎨 Starting Comprehensive 3D Visualization...")
            comprehensive_viz_results = create_comprehensive_visualization_report(
                all_roi_matrices=all_roi_matrices,
                all_labels=all_labels, 
                electrode_locations_valid=electrode_locations_for_viz,
                brodmann_labels_valid=brodmann_labels_for_viz,
                common_rois=config.COMMON_ROIS,
                save_plots=True,
                output_dir="connectome_plots"
            )
            
            if comprehensive_viz_results is not None:
                # Add visualization results to save dictionary
                save_dict['visualization_results'] = comprehensive_viz_results
                print(f"   ✅ Comprehensive 3D visualization completed successfully")
                
                # Display detailed visualization summary
                summary_stats = comprehensive_viz_results['summary_stats']
                diff_analysis = comprehensive_viz_results['diff_analysis']
                
                print(f"\n📊 Detailed Visualization Summary:")
                print(f"   • Hand trials: {summary_stats['n_hand_trials']}")
                print(f"   • Tongue trials: {summary_stats['n_tongue_trials']}")
                print(f"   • Overall hand connectivity: {summary_stats['overall_hand_strength']:.4f}")
                print(f"   • Overall tongue connectivity: {summary_stats['overall_tongue_strength']:.4f}")
                print(f"   • Hand/Tongue ratio: {summary_stats['strength_ratio']:.2f}")
                print(f"   • Mean connectivity difference: {diff_analysis['mean_diff']:.4f}")
                print(f"   • Max hand > tongue: {diff_analysis['max_pos_diff']:+.4f}")
                print(f"   • Max tongue > hand: {diff_analysis['max_neg_diff']:+.4f}")
                print(f"   • Circle plot: {'✅ Created' if summary_stats['circle_plot_created'] else '❌ Failed'}")
                print(f"   • Static plots saved to: connectome_plots/")
            else:
                print(f"   ❌ Comprehensive visualization failed")
            
        except Exception as e:
            print(f"   ⚠️ Comprehensive 3D visualization failed: {e}")
            print(f"      Analysis results will still be saved without visualization")
            import traceback
            traceback.print_exc()
    
    else:
        print(f"   ⚠️ No electrode location data available for 3D visualization")
        print(f"      Creating basic connectivity visualizations instead...")
        
        try:
            basic_viz_results = create_basic_connectivity_visualization(
                all_roi_matrices=all_roi_matrices,
                all_labels=all_labels,
                common_rois=config.COMMON_ROIS,
                save_plots=True,
                output_dir="connectome_plots"
            )
            
            if basic_viz_results is not None:
                save_dict['visualization_results'] = basic_viz_results
                print(f"   ✅ Basic visualization completed successfully")
                
                # Display basic visualization summary
                summary_stats = basic_viz_results['summary_stats']
                print(f"\n📊 Basic Visualization Summary:")
                print(f"   • Hand trials: {summary_stats['n_hand_trials']}")
                print(f"   • Tongue trials: {summary_stats['n_tongue_trials']}")
                print(f"   • Overall hand connectivity: {summary_stats['overall_hand_strength']:.4f}")
                print(f"   • Overall tongue connectivity: {summary_stats['overall_tongue_strength']:.4f}")
                print(f"   • Hand/Tongue ratio: {summary_stats['strength_ratio']:.2f}")
                print(f"   • Mean connectivity difference: {summary_stats['mean_diff']:.4f}")
                print(f"   • Static plots saved to: connectome_plots/")
            else:
                print(f"   ❌ Basic visualization failed")
                
        except Exception as e:
            print(f"   ⚠️ Basic visualization failed: {e}")
            print(f"      Analysis results will still be saved without visualization")
            import traceback
            traceback.print_exc()
        
        print(f"      Consider adding 'elec_xyz' or 'electrode_locations' to your data for 3D visualization")
        print(f"      Available data keys: {list(all_subject_info[0].keys()) if all_subject_info else 'None'}")
    
    save_results(save_dict, "motor_imagery_analysis_results.npz")
    
    # Additional visualization options
    print(f"\n🎨 Additional Visualization Options:")
    print(f"   • Interactive 3D plots are available in the returned objects")
    print(f"   • Static plots saved to: connectome_plots/")
    print(f"   • To create custom visualizations, use the functions from visualization_3d.py")
    
    # Example of how to access visualization results
    if 'visualization_results' in save_dict:
        viz_data = save_dict['visualization_results']
        print(f"\n📋 Available Visualization Data:")
        print(f"   • ROI matrices (hand/tongue): {viz_data['roi_matrix_hand'].shape}")
        print(f"   • ROI names: {viz_data['roi_names']}")
        print(f"   • Summary statistics: connectivity strengths, ratios")
        
        # Check if it's comprehensive (3D) or basic visualization
        if 'main_visualization' in viz_data:
            # Comprehensive 3D visualization
            print(f"   • ROI coordinates: {viz_data['roi_coords'].shape}")
            print(f"   • Interactive plots: hand, tongue, difference")
            print(f"   • Circle plot: {'Available' if viz_data['summary_stats']['circle_plot_created'] else 'Not available'}")
            
            # Show how to access the plots
            print(f"\n💡 To display 3D plots, use:")
            print(f"   • viz_data['main_visualization']['plots']['hand']  # Hand imagery connectome")
            print(f"   • viz_data['main_visualization']['plots']['tongue']  # Tongue imagery connectome")
            print(f"   • viz_data['main_visualization']['plots']['difference']  # Difference connectome")
            print(f"   • viz_data['comparison_plots']  # Additional comparison plots")
            print(f"   • viz_data['summary_stats']  # Connectivity statistics")
            print(f"   • viz_data['diff_analysis']  # Difference analysis results")
        else:
            # Basic visualization
            print(f"   • Heatmaps: hand, tongue, difference")
            print(f"   • Bar plots: overall and ROI-wise strength comparison")
            
            # Show how to access the plots
            print(f"\n💡 To access basic visualization data, use:")
            print(f"   • viz_data['roi_matrix_hand']  # Hand connectivity matrix")
            print(f"   • viz_data['roi_matrix_tongue']  # Tongue connectivity matrix")
            print(f"   • viz_data['diff_matrix']  # Difference matrix")
            print(f"   • viz_data['summary_stats']  # Connectivity statistics")
    
    print(f"\n🎉 Analysis completed successfully!")
    print(f"   • Final test accuracy: {results['test_accuracy']:.3f}")
    print(f"   • Nested CV accuracy: {results['nested_cv']['mean_score']:.3f} ± {results['nested_cv']['std_score']:.3f}")
    
    # Determine visualization status
    if 'visualization_results' in save_dict:
        viz_data = save_dict['visualization_results']
        if 'main_visualization' in viz_data:
            viz_status = "✅ 3D Visualization Completed"
        else:
            viz_status = "✅ Basic Visualization Completed"
    else:
        viz_status = "❌ Not available"
    
    print(f"   • Visualization: {viz_status}")
    print(f"   • Results saved to: motor_imagery_analysis_results.npz")
    print(f"   • Plots saved to: connectome_plots/")
    

if __name__ == "__main__":
    main()