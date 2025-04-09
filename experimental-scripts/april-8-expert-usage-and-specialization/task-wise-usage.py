"""
Task-wise Expert Usage: Partition the evaluation data by task type (e.g. math vs. coding vs. QA, etc., as labeled in Tülu). 
For each model, compute how often each expert is activated for each task subset. 
This yields a matrix of usage frequencies (tasks × 64 experts). 
We will normalize these to get probability distributions per task of which experts are used.
"""

import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle

# Paths to the model data
model_paths = {
    'Model A (teacher)': 'outputs/profiling/moonlight-base/router_tokens.pt',
    'Model B (SFTed)': 'outputs/profiling/olmoe-sft-original-10000/router_tokens.pt',
    'Model C (Distilled)': 'outputs/profiling/olmoe-sft-distill-10000/router_tokens.pt'
}

# Output directory
output_dir = 'experimental-scripts/april-8-expert-usage-and-specialization/outputs'
os.makedirs(output_dir, exist_ok=True)

def compute_task_wise_expert_usage(model_data):
    """Compute task-wise expert usage for each layer in a single model."""
    sources = model_data['sources']
    input_ids = model_data['input_ids']
    
    # Group by source/task
    task_groups = {}
    for i, source in enumerate(sources):
        task_name = source.split('/')[0] if '/' in source else source
        if task_name not in task_groups:
            task_groups[task_name] = []
        task_groups[task_name].append(i)
    
    # Extract model layers with MoE modules
    moe_layers = [key for key in model_data.keys() if 'model.layers' in key and '.mlp' in key]
    
    # Determine number of experts based on the shape of selected_experts
    first_layer = moe_layers[0]
    num_experts = model_data[first_layer]['selected_experts'][0].max().item() + 1
    
    # For each layer and task, compute expert usage
    layer_task_expert_usage = {}
    
    for layer in moe_layers:
        layer_name = layer.replace('model.layers.', 'layer_').replace('.mlp', '')
        task_expert_usage = {}
        
        for task_name, sample_indices in task_groups.items():
            # Initialize counters for this task and layer
            expert_counts = np.zeros(num_experts)
            total_tokens = 0
            
            # Iterate through all samples for this task
            for idx in sample_indices:
                selected_experts = model_data[layer]['selected_experts'][idx]
                # Count occurrences of each expert
                for experts_per_token in selected_experts:
                    for expert_idx in experts_per_token:
                        expert_counts[expert_idx.item()] += 1
                    total_tokens += 1
            
            # Normalize to get probability distribution
            if total_tokens > 0:
                expert_probs = expert_counts / (total_tokens * selected_experts.shape[1])  # account for top-k
                task_expert_usage[task_name] = expert_probs
        
        layer_task_expert_usage[layer_name] = task_expert_usage
    
    return layer_task_expert_usage, task_groups

def main():
    results = {}
    
    # Process each model
    for model_name, model_path in model_paths.items():
        print(f"Processing {model_name}...")
        model_data = torch.load(model_path)
        layer_task_expert_usage, task_groups = compute_task_wise_expert_usage(model_data)
        
        results[model_name] = {
            'layer_task_expert_usage': layer_task_expert_usage,
            'task_groups': task_groups
        }
    
    # Save results
    output_path = os.path.join(output_dir, 'task_wise_expert_usage_per_layer.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()