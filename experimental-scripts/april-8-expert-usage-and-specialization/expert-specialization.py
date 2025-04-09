"""
For a more direct measure of specialization, we can compute for each expert an entropy of its task distribution (does each expert focus on one task or many?). 
We expect specialized experts to have low entropy over task categories (meaning an expert mainly handles one type of task). 
We will compare the distribution of these per-expert entropies for A, B, C. 
Another metric is KL divergence or Jensen-Shannon divergence between the task distribution of each expert in A vs the closest matching expert in C or B (to see if C has experts that function like A's experts).
"""

import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

# Paths to the model data
model_paths = {
    'Model A (teacher)': 'outputs/profiling/moonlight-base/router_tokens.pt',
    'Model B (SFTed)': 'outputs/profiling/olmoe-sft-original-10000/router_tokens.pt',
    'Model C (Distilled)': 'outputs/profiling/olmoe-sft-distill-10000/router_tokens.pt'
}

# Output directory
output_dir = 'experimental-scripts/april-8-expert-usage-and-specialization/outputs'
os.makedirs(output_dir, exist_ok=True)

def compute_expert_task_distributions(model_data):
    """Compute the distribution of tasks for each expert in each layer."""
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
    
    layer_expert_task_distributions = {}
    
    for layer in moe_layers:
        layer_name = layer.replace('model.layers.', 'layer_').replace('.mlp', '')
        
        # Initialize task counts for each expert in this layer
        expert_task_counts = [defaultdict(int) for _ in range(num_experts)]
        expert_total_counts = np.zeros(num_experts)
        
        # Count task occurrences for each expert in this layer
        for task_name, sample_indices in task_groups.items():
            for idx in sample_indices:
                selected_experts = model_data[layer]['selected_experts'][idx]
                for experts_per_token in selected_experts:
                    for expert_idx in experts_per_token:
                        expert_task_counts[expert_idx.item()][task_name] += 1
                        expert_total_counts[expert_idx.item()] += 1
        
        # Convert to probability distributions
        expert_task_distributions = []
        for expert_idx in range(num_experts):
            if expert_total_counts[expert_idx] > 0:
                dist = {task: count / expert_total_counts[expert_idx] 
                       for task, count in expert_task_counts[expert_idx].items()}
            else:
                dist = {}
            expert_task_distributions.append(dist)
        
        layer_expert_task_distributions[layer_name] = expert_task_distributions
    
    return layer_expert_task_distributions, list(task_groups.keys())

def compute_expert_entropies(expert_task_distributions, all_tasks):
    """Compute entropy of task distribution for each expert."""
    expert_entropies = []
    
    for expert_dist in expert_task_distributions:
        # Convert to array with zeros for missing tasks
        dist_array = np.zeros(len(all_tasks))
        for i, task in enumerate(all_tasks):
            dist_array[i] = expert_dist.get(task, 0)
        
        # Skip if expert is not used
        if np.sum(dist_array) == 0:
            expert_entropies.append(0)
            continue
        
        # Normalize if needed
        dist_array = dist_array / np.sum(dist_array)
        
        # Compute entropy
        expert_entropies.append(entropy(dist_array))
    
    return np.array(expert_entropies)

def find_closest_experts(expert_dist_A, expert_dist_B, all_tasks):
    """Find closest expert in B for each expert in A using Jensen-Shannon divergence."""
    num_experts_A = len(expert_dist_A)
    num_experts_B = len(expert_dist_B)
    
    # Initialize distance matrix
    distances = np.zeros((num_experts_A, num_experts_B))
    
    # Compute JS divergence between each pair of experts
    for i in range(num_experts_A):
        dist_A = np.zeros(len(all_tasks))
        for j, task in enumerate(all_tasks):
            dist_A[j] = expert_dist_A[i].get(task, 0)
        
        # Skip if expert is not used
        if np.sum(dist_A) == 0:
            distances[i, :] = np.inf
            continue
        
        # Normalize
        dist_A = dist_A / np.sum(dist_A)
        
        for j in range(num_experts_B):
            dist_B = np.zeros(len(all_tasks))
            for k, task in enumerate(all_tasks):
                dist_B[k] = expert_dist_B[j].get(task, 0)
            
            # Skip if expert is not used
            if np.sum(dist_B) == 0:
                distances[i, j] = np.inf
                continue
            
            # Normalize
            dist_B = dist_B / np.sum(dist_B)
            
            # Compute Jensen-Shannon divergence
            distances[i, j] = jensenshannon(dist_A, dist_B)
    
    # Find closest expert in B for each expert in A
    closest_experts = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)
    
    return closest_experts, min_distances

def main():
    # Load and process each model
    model_data = {}
    layer_expert_task_distributions = {}
    all_tasks = set()
    
    for model_name, model_path in model_paths.items():
        print(f"Processing {model_name}...")
        model_data[model_name] = torch.load(model_path)
        layer_dists, tasks = compute_expert_task_distributions(model_data[model_name])
        layer_expert_task_distributions[model_name] = layer_dists
        all_tasks.update(tasks)
    
    all_tasks = sorted(list(all_tasks))
    
    # Process each layer
    results = {'all_tasks': all_tasks}
    
    # Get all layer names from one model
    sample_model_name = list(model_paths.keys())[0]
    layer_names = list(layer_expert_task_distributions[sample_model_name].keys())
    
    for layer_name in layer_names:
        print(f"Analyzing {layer_name}...")
        layer_results = {}
        
        # Compute expert entropies for each model for this layer
        expert_entropies = {}
        for model_name in model_paths.keys():
            if layer_name in layer_expert_task_distributions[model_name]:
                expert_entropies[model_name] = compute_expert_entropies(
                    layer_expert_task_distributions[model_name][layer_name], all_tasks)
        
        # Find closest experts from B and C to A for this layer
        if layer_name in layer_expert_task_distributions['Model A (teacher)'] and \
           layer_name in layer_expert_task_distributions['Model B (SFTed)'] and \
           layer_name in layer_expert_task_distributions['Model C (Distilled)']:
            
            closest_experts_B_to_A, distances_B_to_A = find_closest_experts(
                layer_expert_task_distributions['Model A (teacher)'][layer_name],
                layer_expert_task_distributions['Model B (SFTed)'][layer_name],
                all_tasks
            )
            
            closest_experts_C_to_A, distances_C_to_A = find_closest_experts(
                layer_expert_task_distributions['Model A (teacher)'][layer_name],
                layer_expert_task_distributions['Model C (Distilled)'][layer_name],
                all_tasks
            )
            
            layer_results['expert_entropies'] = expert_entropies
            layer_results['closest_experts_B_to_A'] = closest_experts_B_to_A
            layer_results['distances_B_to_A'] = distances_B_to_A
            layer_results['closest_experts_C_to_A'] = closest_experts_C_to_A
            layer_results['distances_C_to_A'] = distances_C_to_A
            
            # Print summary statistics for this layer
            print(f"  Average Expert Entropy (lower means more specialized):")
            for model_name, entropies in expert_entropies.items():
                # Filter out unused experts (entropy=0)
                active_entropies = entropies[entropies > 0]
                if len(active_entropies) > 0:
                    print(f"    {model_name}: {np.mean(active_entropies):.4f}")
            
            print(f"  Average JS Distance to Model A experts:")
            if np.any(~np.isinf(distances_B_to_A)):
                print(f"    Model B to A: {np.mean(distances_B_to_A[~np.isinf(distances_B_to_A)]):.4f}")
            if np.any(~np.isinf(distances_C_to_A)):
                print(f"    Model C to A: {np.mean(distances_C_to_A[~np.isinf(distances_C_to_A)]):.4f}")
        
        results[layer_name] = layer_results
    
    # Save results
    output_path = os.path.join(output_dir, 'expert_specialization_per_layer.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()