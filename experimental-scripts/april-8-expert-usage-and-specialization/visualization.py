"""
We will create a heatmap of expert usage by task for each model (experts on one axis, task categories on the other, color = % of that task's tokens routed to that expert). 
This allows visual comparison of specialization patterns. 
For example, in model A we might see distinct columns (experts) brightly associated with certain tasks (indicating that expert handles mostly that task). 
We will see if model C's heatmap looks qualitatively closer to A's than B's does. 
We'll also plot the overall expert usage distribution as bar charts for each model, perhaps side by side, and overlay the distribution for different datasets (Tülu vs calibration) to confirm consistency. 
Additionally, we'll report the numerical dispatch entropy ￼ for each model (and per-task entropies) in a table.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.gridspec as gridspec

# Add Times New Roman font for professional paper-style figures
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Times New Roman'
else:
    print("Times New Roman font not found, using system default serif font")
    plt.rcParams['font.family'] = 'serif'

# Set global plot parameters for professional figures
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#231F20'  # MIT dark gray
plt.rcParams['axes.labelcolor'] = '#231F20'
plt.rcParams['text.color'] = '#231F20'
plt.rcParams['xtick.color'] = '#231F20'
plt.rcParams['ytick.color'] = '#231F20'
plt.rcParams['grid.color'] = '#C2C0BF'  # MIT light silver gray
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# Define MIT colors
MIT_RED = '#A31F34'
MIT_LIGHT_GRAY = '#8A8B8C'
MIT_DARK_GRAY = '#231F20'
MIT_SILVER_GRAY = '#C2C0BF'
MIT_BLACK = '#000000'
MIT_WHITE = '#FFFFFF'

# Create a custom colormap based on MIT colors
MIT_cmap = LinearSegmentedColormap.from_list('MIT_cmap', ['#FFFFFF', '#A31F34'])

# Output directory
output_dir = 'experimental-scripts/april-8-expert-usage-and-specialization/outputs'
os.makedirs(output_dir, exist_ok=True)

# Create directories for layer-specific visualizations
layerwise_dir = os.path.join(output_dir, 'layerwise')
os.makedirs(layerwise_dir, exist_ok=True)

# Define model names for better labeling
model_names = {
    'Model A (teacher)': 'Model A (Teacher)',
    'Model B (SFTed)': 'Model B (SFT)',
    'Model C (Distilled)': 'Model C (Distilled)'
}

# Color mapping for the models
model_colors = {
    'Model A (teacher)': MIT_RED,
    'Model B (SFTed)': MIT_DARK_GRAY,
    'Model C (Distilled)': MIT_LIGHT_GRAY
}

def plot_task_expert_heatmaps(task_expert_usage_data):
    """Create heatmaps of expert usage by task for each model and layer."""
    for model_name, model_data in task_expert_usage_data.items():
        layer_task_expert_usage = model_data['layer_task_expert_usage']
        
        # Create a directory for this model's heatmaps
        model_heatmap_dir = os.path.join(layerwise_dir, f"{model_name.replace(' ', '_')}_heatmaps")
        os.makedirs(model_heatmap_dir, exist_ok=True)
        
        # Create a figure for a model summary (showing a select few layers)
        middle_layers = sorted(list(layer_task_expert_usage.keys()))
        if len(middle_layers) > 4:
            # Select first, last, and two middle layers for summary
            summary_layers = [middle_layers[0], 
                            middle_layers[len(middle_layers)//3], 
                            middle_layers[2*len(middle_layers)//3], 
                            middle_layers[-1]]
        else:
            summary_layers = middle_layers
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()
        
        # Process each layer
        for layer_name, task_expert_usage in layer_task_expert_usage.items():
            # Skip if task_expert_usage is empty
            if not task_expert_usage:
                continue
                
            # Create a DataFrame for the heatmap
            tasks = list(task_expert_usage.keys())
            num_experts = len(next(iter(task_expert_usage.values())))
            
            # Convert to a matrix
            heatmap_data = np.zeros((len(tasks), num_experts))
            for i, task in enumerate(tasks):
                heatmap_data[i, :] = task_expert_usage[task]
            
            # Create DataFrame for better visualization
            df = pd.DataFrame(heatmap_data, index=tasks, columns=[f'Expert {i}' for i in range(num_experts)])
            
            # Sort tasks and experts to make patterns more visible
            # First, sort tasks by their max expert usage
            task_max_expert = np.argmax(heatmap_data, axis=1)
            sorted_task_indices = np.argsort(task_max_expert)
            
            # Create a new sorted DataFrame
            df_sorted = df.iloc[sorted_task_indices]
            
            # Select top experts for better visibility (e.g., top 32)
            top_experts = 32
            expert_usage_sum = df_sorted.sum().nlargest(top_experts).index
            df_sorted = df_sorted[expert_usage_sum]
            
            # Create individual heatmap for each layer
            fig_layer, ax_layer = plt.subplots(figsize=(14, 10))
            
            # Create a custom colormap from white to MIT red
            heatmap = sns.heatmap(df_sorted, cmap=MIT_cmap, norm=LogNorm(), 
                            cbar_kws={'label': 'Token Routing Probability (log scale)', 
                                    'shrink': 0.8,
                                    'pad': 0.01},
                            ax=ax_layer)
            
            # Rotate x tick labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Add professional border around the plot
            for spine in ax_layer.spines.values():
                spine.set_visible(True)
                spine.set_color(MIT_DARK_GRAY)
                spine.set_linewidth(1.5)
            
            plt.title(f'{model_names[model_name]}: {layer_name} - Expert Usage by Task', fontsize=16, pad=20)
            plt.xlabel('Expert ID', fontsize=14, labelpad=10)
            plt.ylabel('Task Type', fontsize=14, labelpad=10)
            
            # Add text annotation for paper-like description
            fig_layer.text(0.5, 0.01, 
                    'Note: Brighter red indicates higher probability of the expert being selected for that task type.', 
                    ha='center', fontsize=10, color=MIT_DARK_GRAY, style='italic')
            
            plt.tight_layout()
            plt.savefig(os.path.join(model_heatmap_dir, f'{layer_name}_task_expert_heatmap.png'))
            plt.close(fig_layer)
            
            # If this is one of the summary layers, add to the summary figure
            if layer_name in summary_layers:
                idx = summary_layers.index(layer_name)
                if idx < len(axes):
                    ax = axes[idx]
                    sns.heatmap(df_sorted, cmap=MIT_cmap, norm=LogNorm(), 
                                cbar_kws={'label': 'Probability (log)', 'shrink': 0.6},
                                ax=ax)
                    ax.set_title(f'{layer_name}', fontsize=12)
                    ax.set_xlabel('Expert ID', fontsize=10)
                    ax.set_ylabel('Task Type', fontsize=10)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Adjust the summary figure
        plt.suptitle(f'{model_names[model_name]}: Expert Usage by Task (Selected Layers)', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save the summary figure
        plt.savefig(os.path.join(layerwise_dir, f'{model_name.replace(" ", "_")}_layer_summary_heatmaps.png'))
        plt.close(fig)

def plot_overall_expert_usage(overall_dist_data):
    """Plot overall expert usage distribution for each model and layer."""
    # Create a directory for distribution plots
    distrib_dir = os.path.join(layerwise_dir, 'distributions')
    os.makedirs(distrib_dir, exist_ok=True)
    
    # Get all unique layer names
    all_layers = set()
    for model_data in overall_dist_data.values():
        all_layers.update(model_data['layer_distributions'].keys())
    
    # Sort layer names
    all_layers = sorted(list(all_layers))
    
    # Create comparison plots for each layer
    for layer_name in all_layers:
        plt.figure(figsize=(10, 6))
        
        for model_name, model_data in overall_dist_data.items():
            if layer_name in model_data['layer_distributions']:
                expert_probs = model_data['layer_distributions'][layer_name]['expert_probs']
                
                # Sort experts by usage for better visualization
                sorted_indices = np.argsort(-expert_probs)
                sorted_probs = expert_probs[sorted_indices]
                
                # Limit to top 32 experts for readability
                top_n = 32
                plt.plot(range(top_n), sorted_probs[:top_n], marker='o', markersize=5,
                       label=f"{model_names[model_name]} (Entropy: {model_data['layer_distributions'][layer_name]['distribution_entropy']:.4f})", 
                       linewidth=2, alpha=0.85, 
                       color=model_colors[model_name], markeredgecolor=MIT_DARK_GRAY)
        
        plt.title(f'{layer_name}: Expert Usage Distribution Comparison', fontsize=16, pad=15)
        plt.xlabel('Expert Rank (sorted by usage)', fontsize=14)
        plt.ylabel('Usage Probability', fontsize=14)
        
        # Add a dashed line indicating uniform distribution
        uniform_prob = 1.0 / 64  # Assuming 64 experts
        plt.axhline(y=uniform_prob, color=MIT_DARK_GRAY, linestyle='--', alpha=0.5, 
                   label=f'Uniform (1/64)')
        
        plt.grid(alpha=0.3, linestyle='--')
        plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(distrib_dir, f'{layer_name}_expert_usage_comparison.png'))
        plt.close()
    
    # Create a multi-layer comparison for each model
    for model_name, model_data in overall_dist_data.items():
        # Select a subset of layers to avoid overcrowding
        layer_names = sorted(list(model_data['layer_distributions'].keys()))
        
        if len(layer_names) > 5:
            # Choose representative layers: early, middle, late layers
            selected_layers = [
                layer_names[0],
                layer_names[len(layer_names)//4],
                layer_names[len(layer_names)//2],
                layer_names[3*len(layer_names)//4],
                layer_names[-1]
            ]
        else:
            selected_layers = layer_names
        
        plt.figure(figsize=(12, 7))
        
        for layer_name in selected_layers:
            if layer_name in model_data['layer_distributions']:
                expert_probs = model_data['layer_distributions'][layer_name]['expert_probs']
                
                # Sort experts by usage for better visualization
                sorted_indices = np.argsort(-expert_probs)
                sorted_probs = expert_probs[sorted_indices]
                
                # Limit to top 32 experts for readability
                top_n = 32
                plt.plot(range(top_n), sorted_probs[:top_n], marker='o', markersize=4,
                       label=f"{layer_name} (Entropy: {model_data['layer_distributions'][layer_name]['distribution_entropy']:.4f})", 
                       linewidth=1.5, alpha=0.8)
        
        plt.title(f'{model_names[model_name]}: Expert Usage Across Layers', fontsize=16, pad=15)
        plt.xlabel('Expert Rank (sorted by usage)', fontsize=14)
        plt.ylabel('Usage Probability', fontsize=14)
        
        # Add a dashed line indicating uniform distribution
        uniform_prob = 1.0 / 64  # Assuming 64 experts
        plt.axhline(y=uniform_prob, color=MIT_DARK_GRAY, linestyle='--', alpha=0.5, 
                   label=f'Uniform (1/64)')
        
        plt.grid(alpha=0.3, linestyle='--')
        plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=9, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(distrib_dir, f'{model_name.replace(" ", "_")}_multi_layer_expert_usage.png'))
        plt.close()

def plot_expert_specialization(expert_spec_data):
    """Plot expert specialization metrics for each layer."""
    # Create a directory for specialization plots
    spec_dir = os.path.join(layerwise_dir, 'specialization')
    os.makedirs(spec_dir, exist_ok=True)
    
    # Extract all layer names
    layer_names = [key for key in expert_spec_data.keys() if key != 'all_tasks']
    
    # JS distance summary (comparing Model B vs C to A)
    js_distances_B = []
    js_distances_C = []
    layer_labels = []
    
    for layer_name in layer_names:
        layer_data = expert_spec_data[layer_name]
        
        if 'distances_B_to_A' not in layer_data or 'distances_C_to_A' not in layer_data:
            continue
            
        # Plot expert entropy distributions for this layer
        plt.figure(figsize=(10, 6))
        
        for model_name, entropies in layer_data['expert_entropies'].items():
            # Filter out unused experts (entropy=0)
            active_entropies = entropies[entropies > 0]
            if len(active_entropies) == 0:
                continue
                
            mean_entropy = np.mean(active_entropies)
            
            # Create histogram with custom style
            plt.hist(active_entropies, bins=20, alpha=0.7, 
                   label=f'{model_names[model_name]} (μ={mean_entropy:.3f})',
                   color=model_colors[model_name], edgecolor=MIT_DARK_GRAY, linewidth=0.5)
            
            # Add a vertical line for the mean entropy
            plt.axvline(x=mean_entropy, color=model_colors[model_name], linestyle='--', linewidth=2, alpha=0.7)
        
        plt.title(f'{layer_name}: Distribution of Expert Task Specialization', fontsize=16, pad=15)
        plt.xlabel('Entropy (lower = more specialized experts)', fontsize=14)
        plt.ylabel('Number of Experts', fontsize=14)
        
        # Style the legend
        legend = plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
        if legend:
            frame = legend.get_frame()
            frame.set_edgecolor(MIT_DARK_GRAY)
        
        plt.tight_layout()
        plt.savefig(os.path.join(spec_dir, f'{layer_name}_entropy_distribution.png'))
        plt.close()
        
        # Plot JS distances for this layer
        distances_B = layer_data['distances_B_to_A']
        distances_C = layer_data['distances_C_to_A']
        
        # Filter out inf values
        valid_B = ~np.isinf(distances_B)
        valid_C = ~np.isinf(distances_C)
        
        if np.any(valid_B) and np.any(valid_C):
            dist_B = distances_B[valid_B]
            dist_C = distances_C[valid_C]
            
            mean_B = np.mean(dist_B)
            mean_C = np.mean(dist_C)
            
            # Add to summary data
            js_distances_B.append(mean_B)
            js_distances_C.append(mean_C)
            layer_labels.append(layer_name)
            
            plt.figure(figsize=(10, 6))
            
            # Create histograms
            plt.hist(dist_B, bins=20, alpha=0.7, 
                   label=f'Model B to A (μ={mean_B:.3f})', 
                   color=model_colors['Model B (SFTed)'], 
                   edgecolor=MIT_DARK_GRAY, linewidth=0.5)
            
            plt.hist(dist_C, bins=20, alpha=0.7, 
                   label=f'Model C to A (μ={mean_C:.3f})', 
                   color=model_colors['Model C (Distilled)'], 
                   edgecolor=MIT_DARK_GRAY, linewidth=0.5)
            
            # Add vertical lines for means
            plt.axvline(x=mean_B, color=model_colors['Model B (SFTed)'], linestyle='--', linewidth=2, alpha=0.7)
            plt.axvline(x=mean_C, color=model_colors['Model C (Distilled)'], linestyle='--', linewidth=2, alpha=0.7)
            
            # Add text box highlighting the difference
            diff_pct = ((mean_B - mean_C) / mean_B) * 100
            if diff_pct > 0:
                text = f"Model C is {diff_pct:.2f}% closer to Model A\nthan Model B is to Model A"
            else:
                text = f"Model B is {-diff_pct:.2f}% closer to Model A\nthan Model C is to Model A"
                
            text_box = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor=MIT_SILVER_GRAY)
            plt.text(0.65, 0.85, text, transform=plt.gca().transAxes, fontsize=11,
                   verticalalignment='top', bbox=text_box)
            
            plt.title(f'{layer_name}: Jensen-Shannon Divergence', fontsize=16, pad=15)
            plt.xlabel('JS Distance (lower = more similar to Model A)', fontsize=14)
            plt.ylabel('Number of Expert Pairs', fontsize=14)
            
            # Style the legend
            legend = plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
            if legend:
                frame = legend.get_frame()
                frame.set_edgecolor(MIT_DARK_GRAY)
            
            plt.tight_layout()
            plt.savefig(os.path.join(spec_dir, f'{layer_name}_js_distances.png'))
            plt.close()
    
    # Create JS distance summary plot across layers
    if js_distances_B and js_distances_C:
        plt.figure(figsize=(12, 7))
        
        # Plot lines showing JS distances across layers
        x = np.arange(len(layer_labels))
        plt.plot(x, js_distances_B, 'o-', label='Model B to A', color=model_colors['Model B (SFTed)'], 
                markersize=6, linewidth=2, alpha=0.8)
        plt.plot(x, js_distances_C, 'o-', label='Model C to A', color=model_colors['Model C (Distilled)'], 
                markersize=6, linewidth=2, alpha=0.8)
        
        # Highlight layers where C is closer to A than B is
        for i, (dist_B, dist_C) in enumerate(zip(js_distances_B, js_distances_C)):
            if dist_C < dist_B:
                plt.fill_between([i-0.3, i+0.3], [0, 0], [max(js_distances_B) * 1.1, max(js_distances_B) * 1.1], 
                                color=MIT_RED, alpha=0.1)
        
        # Calculate overall average improvement
        avg_improvement = np.mean([(b - c) / b * 100 for b, c in zip(js_distances_B, js_distances_C)])
        plt.figtext(0.5, 0.01, 
                  f'On average, Model C is {avg_improvement:.2f}% closer to Model A than Model B is across all layers.', 
                  ha='center', fontsize=11, style='italic')
        
        plt.xticks(x, layer_labels, rotation=45, ha='right')
        plt.title('Jensen-Shannon Divergence Across Layers', fontsize=16, pad=15)
        plt.xlabel('Layer', fontsize=14)
        plt.ylabel('Average JS Distance to Model A', fontsize=14)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(spec_dir, 'js_distance_layer_comparison.png'))
        plt.close()

def create_entropy_table(overall_dist_data):
    """Create a table of entropy values for each model and layer."""
    # Get all unique layer names
    all_layers = set()
    for model_data in overall_dist_data.values():
        all_layers.update(model_data['layer_distributions'].keys())
    
    # Sort layer names
    all_layers = sorted(list(all_layers))
    
    # Create DataFrame for entropy values
    entropy_data = []
    
    for layer_name in all_layers:
        layer_row = {'Layer': layer_name}
        
        for model_name, model_data in overall_dist_data.items():
            if layer_name in model_data['layer_distributions']:
                entropy_val = model_data['layer_distributions'][layer_name]['distribution_entropy']
                layer_row[model_names[model_name]] = entropy_val
        
        entropy_data.append(layer_row)
    
    df_entropy = pd.DataFrame(entropy_data)
    
    # Save entropy table
    df_entropy.to_csv(os.path.join(layerwise_dir, 'layer_entropy_table.csv'), index=False)
    
    # Calculate proximity of Model B and C to Model A
    proximity_data = []
    
    for layer_name in all_layers:
        layer_row = {'Layer': layer_name}
        
        # Check if all models have data for this layer
        if all(layer_name in model_data['layer_distributions'] for model_name, model_data in overall_dist_data.items()):
            entropy_A = overall_dist_data['Model A (teacher)']['layer_distributions'][layer_name]['distribution_entropy']
            entropy_B = overall_dist_data['Model B (SFTed)']['layer_distributions'][layer_name]['distribution_entropy']
            entropy_C = overall_dist_data['Model C (Distilled)']['layer_distributions'][layer_name]['distribution_entropy']
            
            diff_B = abs(entropy_A - entropy_B)
            diff_C = abs(entropy_A - entropy_C)
            
            layer_row['Entropy A'] = entropy_A
            layer_row['Entropy B'] = entropy_B
            layer_row['Entropy C'] = entropy_C
            layer_row['|B-A|'] = diff_B
            layer_row['|C-A|'] = diff_C
            layer_row['Closer to A'] = 'Model C' if diff_C < diff_B else 'Model B'
            layer_row['Improvement (%)'] = ((diff_B - diff_C) / diff_B) * 100 if diff_B > 0 else 0
            
            proximity_data.append(layer_row)
    
    df_proximity = pd.DataFrame(proximity_data)
    
    # Save proximity analysis table
    df_proximity.to_csv(os.path.join(layerwise_dir, 'entropy_proximity_analysis.csv'), index=False)
    
    # Create visualization of the entropy proximity
    if not df_proximity.empty:
        plt.figure(figsize=(12, 7))
        
        layers = df_proximity['Layer'].tolist()
        diff_B = df_proximity['|B-A|'].tolist()
        diff_C = df_proximity['|C-A|'].tolist()
        
        x = np.arange(len(layers))
        
        plt.bar(x - 0.2, diff_B, width=0.4, label='|Model B - Model A|', 
               color=model_colors['Model B (SFTed)'], alpha=0.7)
        plt.bar(x + 0.2, diff_C, width=0.4, label='|Model C - Model A|', 
               color=model_colors['Model C (Distilled)'], alpha=0.7)
        
        # Highlight layers where C is closer to A than B is
        for i, row in enumerate(zip(diff_B, diff_C)):
            if row[1] < row[0]:
                plt.plot(i, 0, 'v', color=MIT_RED, markersize=10)
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.xticks(x, layers, rotation=45, ha='right')
        plt.title('Entropy Distance from Model A', fontsize=16, pad=15)
        plt.xlabel('Layer', fontsize=14)
        plt.ylabel('Absolute Difference in Entropy', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate overall improvement
        avg_improvement = df_proximity['Improvement (%)'].mean()
        plt.figtext(0.5, 0.01, 
                  f'On average, Model C\'s entropy is {avg_improvement:.2f}% closer to Model A\'s than Model B\'s is.', 
                  ha='center', fontsize=11, style='italic')
        
        plt.tight_layout()
        plt.savefig(os.path.join(layerwise_dir, 'entropy_proximity_visualization.png'))
        plt.close()

def main():
    # Load results from previous scripts
    task_wise_path = os.path.join(output_dir, 'task_wise_expert_usage_per_layer.pkl')
    overall_dist_path = os.path.join(output_dir, 'overall_distribution_per_layer.pkl')
    expert_spec_path = os.path.join(output_dir, 'expert_specialization_per_layer.pkl')
    
    # Check if all required files exist
    all_files_exist = all(os.path.exists(p) for p in [task_wise_path, overall_dist_path, expert_spec_path])
    
    if not all_files_exist:
        print("Error: Required data files are missing. Please run the other scripts first.")
        return
    
    print("Loading data files...")
    with open(task_wise_path, 'rb') as f:
        task_wise_data = pickle.load(f)
    
    with open(overall_dist_path, 'rb') as f:
        overall_dist_data = pickle.load(f)
    
    with open(expert_spec_path, 'rb') as f:
        expert_spec_data = pickle.load(f)
    
    print("Creating visualizations...")
    
    # Create heatmaps
    print("Creating task-expert heatmaps...")
    plot_task_expert_heatmaps(task_wise_data)
    
    # Create distribution plots
    print("Creating expert usage distribution plots...")
    plot_overall_expert_usage(overall_dist_data)
    
    # Create expert specialization plots
    print("Creating expert specialization plots...")
    plot_expert_specialization(expert_spec_data)
    
    # Create entropy tables and visualization
    print("Creating entropy tables and visualization...")
    create_entropy_table(overall_dist_data)
    
    print(f"All visualizations and tables saved to {layerwise_dir}")

if __name__ == "__main__":
    main()