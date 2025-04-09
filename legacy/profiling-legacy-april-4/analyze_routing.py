import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import argparse

# MIT color palette
MIT_COLORS = ["#8A1267", "#DE1B1B", "#F15A22", "#FBB03B", "#FCEE23", "#D9E021", "#8CC63F", "#009245", 
              "#0072BC", "#2E3192", "#662D91", "#92278F"]

def load_routing_data(model_dirs):
    data_dict = {}
    for model_name, model_dir in model_dirs.items():
        path = os.path.join(model_dir, "router_tokens.pt")
        print(f"Loading data from {path}")
        data = torch.load(path)
        data_dict[model_name] = data
    return data_dict

def find_common_sequences(data_dict):
    """Find common sequences (input_ids) between models"""
    # Create an empty dictionary for each model
    common_sequences = {model_name: {} for model_name in data_dict.keys()}
    
    models = list(data_dict.keys())
    
    # Compare each pair of models
    for i, model1 in enumerate(models):
        model1_inputs = data_dict[model1]["input_ids"]
        
        for model1_idx, input_seq1 in enumerate(model1_inputs):
            # Record this sequence for the first model
            if model1_idx not in common_sequences[model1]:
                common_sequences[model1][model1_idx] = []
            
            # Check if this sequence exists in other models
            for j, model2 in enumerate(models):
                if model1 == model2:
                    continue
                    
                model2_inputs = data_dict[model2]["input_ids"]
                
                for model2_idx, input_seq2 in enumerate(model2_inputs):
                    if torch.equal(input_seq1, input_seq2):
                        if model2_idx not in common_sequences[model2]:
                            common_sequences[model2][model2_idx] = []
                        
                        # Store the matching indices
                        common_sequences[model1][model1_idx].append((model2, model2_idx))
                        common_sequences[model2][model2_idx].append((model1, model1_idx))
                        break
    
    # Filter to keep only sequences that are found in all models
    for model_name in models:
        to_remove = []
        for seq_idx in common_sequences[model_name]:
            matches = common_sequences[model_name][seq_idx]
            if len(matches) < len(models) - 1:
                to_remove.append(seq_idx)
        
        for seq_idx in to_remove:
            del common_sequences[model_name][seq_idx]
    
    print(f"Found {len(common_sequences[models[0]])} common sequences across all models")
    
    # Convert to a more convenient format for pairwise comparisons
    pairwise_common_seqs = {}
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i == j:
                continue
                
            pair_name = f"{model1}_{model2}"
            pairwise_common_seqs[pair_name] = {}
            
            for seq_idx1 in common_sequences[model1]:
                # Find the matching sequence in model2
                for other_model, other_idx in common_sequences[model1][seq_idx1]:
                    if other_model == model2:
                        pairwise_common_seqs[pair_name][seq_idx1] = other_idx
                        break
    
    return pairwise_common_seqs

def compare_expert_selection(data_dict, common_sequences, layer_indices=None):
    """Compare expert selection between models for common sequences"""
    if layer_indices is None:
        # Use all layers
        layer_indices = range(len([k for k in data_dict[list(data_dict.keys())[0]].keys() 
                               if k.startswith("model.layers") and k.endswith("mlp")]))
    
    results = {}
    models = list(data_dict.keys())
    
    for layer_idx in layer_indices:
        layer_key = f"model.layers.{layer_idx}.mlp"
        layer_results = {}
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models[i+1:], i+1):
                pair_name = f"{model1}_vs_{model2}"
                pair_key = f"{model1}_{model2}"
                
                if pair_key not in common_sequences:
                    continue
                    
                if pair_name not in layer_results:
                    layer_results[pair_name] = {
                        "total_tokens": 0,
                        "exact_match_count": 0,
                        "top1_match_count": 0,
                        "overlap_counts": np.zeros(9),  # 0 to 8 experts in common
                        "position_matches": np.zeros(8),  # Track matches at each position
                    }
                
                # For each common sequence
                for idx1, idx2 in common_sequences[pair_key].items():
                    if (idx1 >= len(data_dict[model1][layer_key]["selected_experts"]) or 
                        idx2 >= len(data_dict[model2][layer_key]["selected_experts"])):
                        continue
                    
                    # Get routing data for this sequence
                    experts1 = data_dict[model1][layer_key]["selected_experts"][idx1]
                    experts2 = data_dict[model2][layer_key]["selected_experts"][idx2]
                    
                    # Ensure same sequence length
                    min_len = min(len(experts1), len(experts2))
                    
                    # Compare token by token
                    for token_idx in range(min_len):
                        # Get experts for this token
                        token_experts1 = experts1[token_idx]
                        token_experts2 = experts2[token_idx]
                        
                        # Count exact matches
                        if torch.equal(token_experts1, token_experts2):
                            layer_results[pair_name]["exact_match_count"] += 1
                        
                        # Count top-1 expert matches
                        if token_experts1[0] == token_experts2[0]:
                            layer_results[pair_name]["top1_match_count"] += 1
                        
                        # Count position-specific matches
                        for pos in range(8):
                            if token_experts1[pos] == token_experts2[pos]:
                                layer_results[pair_name]["position_matches"][pos] += 1
                        
                        # Count overlap
                        overlap = len(set(token_experts1.tolist()) & set(token_experts2.tolist()))
                        layer_results[pair_name]["overlap_counts"][overlap] += 1
                        
                        layer_results[pair_name]["total_tokens"] += 1
        
        results[layer_idx] = layer_results
    
    return results

def visualize_expert_selection_comparison(results, output_dir):
    """Visualize expert selection comparison results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Aggregate results across layers
    aggregated = {}
    
    for layer_idx, layer_results in results.items():
        for pair_name, pair_results in layer_results.items():
            if pair_name not in aggregated:
                aggregated[pair_name] = {
                    "layers": [],
                    "exact_match_ratio": [],
                    "top1_match_ratio": [],
                    "avg_overlap": [],
                    "position_match_ratios": [[] for _ in range(8)],
                }
            
            total = pair_results["total_tokens"]
            if total > 0:
                aggregated[pair_name]["layers"].append(layer_idx)
                aggregated[pair_name]["exact_match_ratio"].append(pair_results["exact_match_count"] / total)
                aggregated[pair_name]["top1_match_ratio"].append(pair_results["top1_match_count"] / total)
                
                # Calculate average overlap
                weighted_sum = sum(i * count for i, count in enumerate(pair_results["overlap_counts"]))
                avg_overlap = weighted_sum / total
                aggregated[pair_name]["avg_overlap"].append(avg_overlap)
                
                # Calculate position match ratios
                for pos in range(8):
                    pos_ratio = pair_results["position_matches"][pos] / total
                    aggregated[pair_name]["position_match_ratios"][pos].append(pos_ratio)
    
    # Plot exact match ratio across layers
    plt.figure(figsize=(12, 8))
    for pair_name, pair_data in aggregated.items():
        plt.plot(pair_data["layers"], pair_data["exact_match_ratio"], 
                 label=pair_name, marker='o', linewidth=2)
    
    plt.title("Exact Expert Selection Match Ratio Across Layers", fontsize=16)
    plt.xlabel("Layer Index", fontsize=14)
    plt.ylabel("Exact Match Ratio", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exact_match_ratio.png"), dpi=300)
    
    # Plot top-1 match ratio across layers
    plt.figure(figsize=(12, 8))
    for pair_name, pair_data in aggregated.items():
        plt.plot(pair_data["layers"], pair_data["top1_match_ratio"], 
                 label=pair_name, marker='o', linewidth=2)
    
    plt.title("Top-1 Expert Selection Match Ratio Across Layers", fontsize=16)
    plt.xlabel("Layer Index", fontsize=14)
    plt.ylabel("Top-1 Match Ratio", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top1_match_ratio.png"), dpi=300)
    
    # Plot average overlap across layers
    plt.figure(figsize=(12, 8))
    for pair_name, pair_data in aggregated.items():
        plt.plot(pair_data["layers"], pair_data["avg_overlap"], 
                 label=pair_name, marker='o', linewidth=2)
    
    plt.title("Average Expert Overlap Across Layers", fontsize=16)
    plt.xlabel("Layer Index", fontsize=14)
    plt.ylabel("Average Number of Experts in Common", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_overlap.png"), dpi=300)
    
    # Plot position-specific match ratios
    plt.figure(figsize=(14, 10))
    bar_width = 0.15
    positions = range(8)
    
    for i, pair_name in enumerate(aggregated.keys()):
        avg_position_ratios = [np.mean(aggregated[pair_name]["position_match_ratios"][pos]) 
                              for pos in range(8)]
        
        offset = (i - len(aggregated) / 2 + 0.5) * bar_width
        plt.bar([p + offset for p in positions], avg_position_ratios, 
                width=bar_width, label=pair_name, color=MIT_COLORS[i % len(MIT_COLORS)])
    
    plt.title("Expert Selection Match Ratio by Position (Averaged Over Layers)", fontsize=16)
    plt.xlabel("Expert Position", fontsize=14)
    plt.ylabel("Match Ratio", fontsize=14)
    plt.xticks(positions, [f"Pos {i+1}" for i in positions], fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "position_match_ratio.png"), dpi=300)

def identify_changed_tokens(data_dict, common_sequences, layer_indices=None, top_n=100):
    """Identify tokens with the most changes in routing patterns"""
    if layer_indices is None:
        # Use all layers
        layer_indices = range(len([k for k in data_dict[list(data_dict.keys())[0]].keys() 
                               if k.startswith("model.layers") and k.endswith("mlp")]))
    
    models = list(data_dict.keys())
    results = {}
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models[i+1:], i+1):
            pair_name = f"{model1}_vs_{model2}"
            pair_key = f"{model1}_{model2}"
            
            if pair_key not in common_sequences:
                continue
                
            results[pair_name] = {
                "token_changes": []  # Will store (sequence_idx, token_idx, layer_idx, overlap)
            }
            
            # For each common sequence
            for idx1, idx2 in common_sequences[pair_key].items():
                if idx1 >= len(data_dict[model1]["input_ids"]):
                    continue
                    
                # Get input_ids
                input_ids1 = data_dict[model1]["input_ids"][idx1]
                
                # For each layer
                for layer_idx in layer_indices:
                    layer_key = f"model.layers.{layer_idx}.mlp"
                    
                    if (idx1 >= len(data_dict[model1][layer_key]["selected_experts"]) or 
                        idx2 >= len(data_dict[model2][layer_key]["selected_experts"])):
                        continue
                        
                    # Get routing data for this sequence
                    experts1 = data_dict[model1][layer_key]["selected_experts"][idx1]
                    experts2 = data_dict[model2][layer_key]["selected_experts"][idx2]
                    
                    # Ensure same sequence length
                    min_len = min(len(experts1), len(experts2))
                    
                    # Compare token by token
                    for token_idx in range(min_len):
                        # Get experts for this token
                        token_experts1 = experts1[token_idx]
                        token_experts2 = experts2[token_idx]
                        
                        # Calculate overlap
                        overlap = len(set(token_experts1.tolist()) & set(token_experts2.tolist()))
                        
                        # Record change information
                        results[pair_name]["token_changes"].append({
                            "sequence_idx": idx1,
                            "token_idx": token_idx,
                            "layer_idx": layer_idx,
                            "token_id": input_ids1[token_idx].item() if token_idx < len(input_ids1) else -1,
                            "overlap": overlap,
                            "experts1": token_experts1.tolist(),
                            "experts2": token_experts2.tolist()
                        })
    
    # Sort tokens by overlap and get top_n changes
    for pair_name in results:
        results[pair_name]["token_changes"].sort(key=lambda x: x["overlap"])
        results[pair_name]["top_changed_tokens"] = results[pair_name]["token_changes"][:top_n]
    
    return results

def analyze_routing_similarity(data_dict, common_sequences, layer_indices=None):
    """Analyze similarity of routing patterns between models"""
    if layer_indices is None:
        # Use all layers
        layer_indices = range(len([k for k in data_dict[list(data_dict.keys())[0]].keys() 
                               if k.startswith("model.layers") and k.endswith("mlp")]))
    
    models = list(data_dict.keys())
    results = {}
    
    for layer_idx in layer_indices:
        layer_key = f"model.layers.{layer_idx}.mlp"
        
        # Create empty matrices to store similarities
        num_models = len(models)
        cos_sim_matrix = np.zeros((num_models, num_models))
        
        # Calculate aggregated routing patterns for each model
        model_patterns = {}
        
        for i, model_name in enumerate(models):
            # Create a matrix counting how many times each token routes to each expert
            # We'll use one-hot encoding and sum across tokens
            num_experts = 64  # Assuming 64 experts per layer for OLMoE
            token_expert_matrix = np.zeros((0, num_experts))
            
            # For each model pair, get common sequences
            for j, other_model in enumerate(models):
                if i == j:
                    continue
                    
                pair_key = f"{model_name}_{other_model}"
                reverse_pair_key = f"{other_model}_{model_name}"
                
                if pair_key in common_sequences:
                    for idx1 in common_sequences[pair_key]:
                        if idx1 >= len(data_dict[model_name][layer_key]["selected_experts"]):
                            continue
                            
                        experts = data_dict[model_name][layer_key]["selected_experts"][idx1]
                        
                        # Convert to one-hot encoding
                        for token_experts in experts:
                            row = np.zeros(num_experts)
                            for expert_idx in token_experts:
                                row[expert_idx] = 1
                            token_expert_matrix = np.vstack((token_expert_matrix, row))
                elif reverse_pair_key in common_sequences:
                    for idx2, idx1 in common_sequences[reverse_pair_key].items():
                        if idx1 >= len(data_dict[model_name][layer_key]["selected_experts"]):
                            continue
                            
                        experts = data_dict[model_name][layer_key]["selected_experts"][idx1]
                        
                        # Convert to one-hot encoding
                        for token_experts in experts:
                            row = np.zeros(num_experts)
                            for expert_idx in token_experts:
                                row[expert_idx] = 1
                            token_expert_matrix = np.vstack((token_expert_matrix, row))
            
            # Aggregate to get a distribution
            if len(token_expert_matrix) > 0:
                model_patterns[model_name] = np.mean(token_expert_matrix, axis=0)
            else:
                # If no data, use zeros
                model_patterns[model_name] = np.zeros(num_experts)
        
        # Calculate pairwise similarities
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                pattern1 = model_patterns[model1]
                pattern2 = model_patterns[model2]
                cos_sim_matrix[i, j] = cosine_similarity([pattern1], [pattern2])[0, 0]
        
        results[layer_idx] = {
            "cos_sim_matrix": cos_sim_matrix,
            "model_patterns": model_patterns
        }
    
    return results, models

def visualize_routing_similarity(similarity_results, models, output_dir):
    """Visualize routing similarity between models"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Aggregate results across layers
    avg_cos_sim_matrix = np.zeros((len(models), len(models)))
    layer_matrices = []
    
    for layer_idx, layer_results in similarity_results.items():
        avg_cos_sim_matrix += layer_results["cos_sim_matrix"]
        layer_matrices.append(layer_results["cos_sim_matrix"])
    
    avg_cos_sim_matrix /= len(similarity_results)
    
    # Plot average cosine similarity heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_cos_sim_matrix, annot=True, fmt=".3f", cmap="YlGnBu",
                xticklabels=models, yticklabels=models)
    plt.title("Average Routing Pattern Similarity Across Layers", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_routing_similarity.png"), dpi=300)
    
    # Plot similarity over layers for each model pair
    plt.figure(figsize=(12, 8))
    
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            pair_name = f"{models[i]}_vs_{models[j]}"
            similarities = [matrix[i, j] for matrix in layer_matrices]
            
            plt.plot(list(similarity_results.keys()), similarities, 
                     label=pair_name, marker='o', linewidth=2)
    
    plt.title("Routing Pattern Similarity Across Layers", fontsize=16)
    plt.xlabel("Layer Index", fontsize=14)
    plt.ylabel("Cosine Similarity", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_routing_similarity.png"), dpi=300)
    
    # Plot distribution patterns for a selected layer
    selected_layer = list(similarity_results.keys())[len(similarity_results) // 2]  # middle layer
    
    plt.figure(figsize=(14, 8))
    bar_width = 0.8 / len(models)
    
    for i, model_name in enumerate(models):
        pattern = similarity_results[selected_layer]["model_patterns"][model_name]
        offset = (i - len(models) / 2 + 0.5) * bar_width
        plt.bar(np.arange(len(pattern)) + offset, pattern, width=bar_width, 
                label=model_name, alpha=0.7, color=MIT_COLORS[i % len(MIT_COLORS)])
    
    plt.title(f"Expert Utilization Pattern for Layer {selected_layer}", fontsize=16)
    plt.xlabel("Expert Index", fontsize=14)
    plt.ylabel("Utilization Rate", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"layer_{selected_layer}_patterns.png"), dpi=300)

def compute_model_weight_similarity(model_paths, output_dir):
    """Compute weight similarity between models"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model state dictionaries
    state_dicts = {}
    for model_name, model_path in model_paths.items():
        print(f"Loading model weights from {model_path}")
        state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location="cpu")
        state_dicts[model_name] = state_dict
    
    # Compute similarities
    models = list(state_dicts.keys())
    num_models = len(models)
    
    # Group weights by layers
    layer_weights = {}
    
    # First, identify common parameters across models
    common_keys = set(state_dicts[models[0]].keys())
    for model in models[1:]:
        common_keys &= set(state_dicts[model].keys())
    
    # Group by layer
    for key in common_keys:
        if "layers" in key:
            parts = key.split(".")
            if len(parts) >= 3:
                layer_idx = int(parts[2])
                if layer_idx not in layer_weights:
                    layer_weights[layer_idx] = []
                layer_weights[layer_idx].append(key)
    
    # Compute similarities by layer
    layer_similarities = {}
    for layer_idx, keys in layer_weights.items():
        cos_sim_matrix = np.zeros((num_models, num_models))
        mse_matrix = np.zeros((num_models, num_models))
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                # Concatenate all parameters for this layer
                params1 = torch.cat([state_dicts[model1][k].flatten() for k in keys])
                params2 = torch.cat([state_dicts[model2][k].flatten() for k in keys])
                
                # Compute cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(params1.unsqueeze(0), params2.unsqueeze(0))[0].item()
                cos_sim_matrix[i, j] = cos_sim
                
                # Compute MSE
                mse = torch.mean((params1 - params2) ** 2).item()
                mse_matrix[i, j] = mse
        
        layer_similarities[layer_idx] = {
            "cos_sim": cos_sim_matrix,
            "mse": mse_matrix
        }
    
    # Plot results
    plt.figure(figsize=(15, 12))
    
    for idx, (title, cmap) in enumerate([("Cosine Similarity", "YlGnBu"), ("Mean Squared Error", "YlOrRd")]):
        plt.subplot(2, 1, idx+1)
        
        # Compute average across layers
        avg_matrix = np.zeros((num_models, num_models))
        metric_key = "cos_sim" if idx == 0 else "mse"
        
        for layer_idx in layer_similarities:
            avg_matrix += layer_similarities[layer_idx][metric_key]
        
        avg_matrix /= len(layer_similarities)
        
        # Plot heatmap
        sns.heatmap(avg_matrix, annot=True, fmt=".4f", cmap=cmap,
                    xticklabels=models, yticklabels=models)
        plt.title(f"Average {title} Across Layers", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "weight_similarity.png"), dpi=300)
    
    # Plot layer-by-layer similarity for each model pair
    plt.figure(figsize=(12, 8))
    
    for i in range(num_models):
        for j in range(i+1, num_models):
            pair_name = f"{models[i]}_vs_{models[j]}"
            similarities = [layer_similarities[layer_idx]["cos_sim"][i, j] 
                           for layer_idx in sorted(layer_similarities.keys())]
            
            plt.plot(sorted(layer_similarities.keys()), similarities, 
                     label=pair_name, marker='o', linewidth=2)
    
    plt.title("Weight Cosine Similarity Across Layers", fontsize=16)
    plt.xlabel("Layer Index", fontsize=14)
    plt.ylabel("Cosine Similarity", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_weight_similarity.png"), dpi=300)
    
    return layer_similarities

def main():
    parser = argparse.ArgumentParser(description="Analyze routing patterns in MoE models")
    parser.add_argument("--output_dir", type=str, default="./outputs/routing_analysis",
                        help="Directory to save analysis results")
    args = parser.parse_args()
    
    # Define model directories
    model_dirs = {
        "olmoe_base": "./outputs/profiling/olmoe-base",
        "olmoe_distill": "./outputs/profiling/olmoe-sft-distill-10000",
        "olmoe_original": "./outputs/profiling/olmoe-sft-original-10000",
        "moonlight": "./outputs/profiling/moonlight-base"
    }
    
    # Define model checkpoint paths for weight similarity
    model_paths = {
        "olmoe_base": "checkpoints/allenai/OLMoE-1B-7B-0125",
        "olmoe_distill": "checkpoints/olmoe-1b-7b-0125-sft-distilled-moonlight-filtered/checkpoint-10000",
        "olmoe_original": "checkpoints/olmoe-1b-7b-0125-sft-original-filtered/checkpoint-10000"
    }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load routing data
    data_dict = load_routing_data(model_dirs)
    
    # Find common sequences
    print("Finding common sequences between models...")
    common_sequences = find_common_sequences(data_dict)
    
    # Compare expert selection
    print("Comparing expert selection between models...")
    selection_results = compare_expert_selection(data_dict, common_sequences)
    
    # Visualize expert selection comparison
    print("Visualizing expert selection comparison...")
    visualize_expert_selection_comparison(selection_results, os.path.join(args.output_dir, "expert_selection"))
    
    # Identify changed tokens
    print("Identifying tokens with largest routing changes...")
    changed_tokens = identify_changed_tokens(data_dict, common_sequences)
    
    # Save changed tokens results
    for pair_name, results in changed_tokens.items():
        with open(os.path.join(args.output_dir, f"{pair_name}_top_changed_tokens.txt"), "w") as f:
            f.write(f"Top changed tokens for {pair_name}:\n")
            for i, token_info in enumerate(results["top_changed_tokens"]):
                f.write(f"{i+1}. Sequence {token_info['sequence_idx']}, Token {token_info['token_idx']}, "
                        f"Layer {token_info['layer_idx']}, Token ID {token_info['token_id']}, "
                        f"Overlap {token_info['overlap']}\n")
                f.write(f"   Model 1 experts: {token_info['experts1']}\n")
                f.write(f"   Model 2 experts: {token_info['experts2']}\n\n")
    
    # Analyze routing similarity
    print("Analyzing routing similarity between models...")
    similarity_results, models = analyze_routing_similarity(data_dict, common_sequences)
    
    # Visualize routing similarity
    print("Visualizing routing similarity...")
    visualize_routing_similarity(similarity_results, models, os.path.join(args.output_dir, "routing_similarity"))
    
    # Compute model weight similarity
    print("Computing model weight similarity...")
    try:
        compute_model_weight_similarity(model_paths, os.path.join(args.output_dir, "weight_similarity"))
    except Exception as e:
        print(f"Error computing weight similarity: {e}")
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 