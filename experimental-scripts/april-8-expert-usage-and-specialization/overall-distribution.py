"""
Overall distribution: Compute each expert's overall fraction of tokens processed. 
From this, calculate the entropy of the distribution of token assignments across the 64 experts for each model. 
A lower entropy (more peaked distribution) indicates a few experts take most of the load (high specialization), whereas a higher entropy indicates more balanced usage (lower specialization). 
We will also compute the entropy per task to see if some tasks are handled by a small subset of experts (which would imply specialization by domain).
"""

import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from scipy.stats import entropy

# Paths to the model data
model_paths = {
    'Model A (teacher)': 'outputs/profiling/moonlight-base/router_tokens.pt',
    'Model B (SFTed)': 'outputs/profiling/olmoe-sft-original-10000/router_tokens.pt',
    'Model C (Distilled)': 'outputs/profiling/olmoe-sft-distill-10000/router_tokens.pt'
}

# Output directory
output_dir = 'experimental-scripts/april-8-expert-usage-and-specialization/outputs'
os.makedirs(output_dir, exist_ok=True)

def compute_overall_distribution(model_data):
    """Compute overall distribution of experts for each layer in a single model."""
    # Extract model layers with MoE modules
    moe_layers = [key for key in model_data.keys() if 'model.layers' in key and '.mlp' in key]
    
    # Determine number of experts based on the shape of selected_experts
    first_layer = moe_layers[0]
    num_experts = model_data[first_layer]['selected_experts'][0].max().item() + 1
    
    layer_results = {}
    
    for layer in moe_layers:
        layer_name = layer.replace('model.layers.', 'layer_').replace('.mlp', '')
        
        # Initialize counters for this layer
        expert_counts = np.zeros(num_experts)
        total_tokens = 0
        
        # Count expert occurrences across all samples for this layer
        for selected_experts_batch in model_data[layer]['selected_experts']:
            for experts_per_token in selected_experts_batch:
                for expert_idx in experts_per_token:
                    expert_counts[expert_idx.item()] += 1
                total_tokens += 1
        
        # Normalize to get probability distribution
        expert_probs = expert_counts / expert_counts.sum() if expert_counts.sum() > 0 else expert_counts
        
        # Calculate entropy of the distribution
        distribution_entropy = entropy(expert_probs)
        
        layer_results[layer_name] = {
            'expert_probs': expert_probs,
            'distribution_entropy': distribution_entropy
        }
    
    return layer_results

def compute_task_wise_entropy(model_data):
    """Compute entropy of expert distribution for each task and each layer."""
    sources = model_data['sources']
    
    # Group by source/task
    task_groups = {}
    for i, source in enumerate(sources):
        task_name = source.split('/')[0] if '/' in source else source
        if task_name not in task_groups:
            task_groups[task_name] = []
        task_groups[task_name].append(i)
    
    # Extract model layers with MoE modules
    moe_layers = [key for key in model_data.keys() if 'model.layers' in key and '.mlp' in key]
    
    # Determine number of experts
    first_layer = moe_layers[0]
    num_experts = model_data[first_layer]['selected_experts'][0].max().item() + 1
    
    layer_task_entropies = {}
    
    for layer in moe_layers:
        layer_name = layer.replace('model.layers.', 'layer_').replace('.mlp', '')
        task_entropies = {}
        
        for task_name, sample_indices in task_groups.items():
            # Initialize counters for this task and layer
            expert_counts = np.zeros(num_experts)
            
            # Iterate through all samples for this task
            for idx in sample_indices:
                selected_experts = model_data[layer]['selected_experts'][idx]
                # Count occurrences of each expert
                for experts_per_token in selected_experts:
                    for expert_idx in experts_per_token:
                        expert_counts[expert_idx.item()] += 1
            
            # Normalize and compute entropy
            expert_probs = expert_counts / expert_counts.sum() if expert_counts.sum() > 0 else expert_counts
            task_entropies[task_name] = entropy(expert_probs)
        
        layer_task_entropies[layer_name] = task_entropies
    
    return layer_task_entropies

def main():
    results = {}
    
    # Process each model
    for model_name, model_path in model_paths.items():
        print(f"Processing {model_name}...")
        model_data = torch.load(model_path)
        
        # Compute overall distribution and entropy for each layer
        layer_distributions = compute_overall_distribution(model_data)
        
        # Compute task-wise entropies for each layer
        layer_task_entropies = compute_task_wise_entropy(model_data)
        
        results[model_name] = {
            'layer_distributions': layer_distributions,
            'layer_task_entropies': layer_task_entropies
        }
    
    # Save results
    output_path = os.path.join(output_dir, 'overall_distribution_per_layer.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Print summary of average entropy across layers
    print("\nAverage Overall Distribution Entropy Across Layers:")
    for model_name, model_results in results.items():
        layer_entropies = [layer_data['distribution_entropy'] for layer_data in model_results['layer_distributions'].values()]
        avg_entropy = np.mean(layer_entropies)
        print(f"{model_name}: {avg_entropy:.4f}")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()