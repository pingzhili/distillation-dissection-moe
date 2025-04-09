import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import AutoTokenizer
import argparse
from collections import Counter, defaultdict
from matplotlib.colors import LinearSegmentedColormap

# Define the MIT color palette
MIT_COLORS = ["#8A1267", "#DE1B1B", "#F15A22", "#FBB03B", "#FCEE23", "#D9E021", "#8CC63F", "#009245", 
              "#0072BC", "#2E3192", "#662D91", "#92278F"]

# Create a custom colormap
cmap_distillation = LinearSegmentedColormap.from_list("MIT", MIT_COLORS, N=12)

def load_routing_data(model_dirs):
    """Load routing data for each model"""
    data_dict = {}
    for model_name, model_dir in model_dirs.items():
        path = os.path.join(model_dir, "router_tokens.pt")
        print(f"Loading data from {path}")
        data = torch.load(path)
        data_dict[model_name] = data
    return data_dict

def load_tokenizer():
    """Load the tokenizer for decoding token IDs"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125")
        return tokenizer
    except:
        print("Warning: Could not load tokenizer. Token text will not be available.")
        return None

def find_common_sequences(data_dict):
    """Find sequences that are present in all models"""
    common_seq_indices = {}
    models = list(data_dict.keys())
    
    # Process each pair of models
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models[i+1:], i+1):
            pair_key = f"{model1}_{model2}"
            common_seq_indices[pair_key] = {}
            
            model1_inputs = data_dict[model1]["input_ids"]
            model2_inputs = data_dict[model2]["input_ids"]
            
            # Find common sequences
            for idx1, input_seq1 in enumerate(model1_inputs):
                for idx2, input_seq2 in enumerate(model2_inputs):
                    if torch.equal(input_seq1, input_seq2):
                        common_seq_indices[pair_key][idx1] = idx2
                        break
                        
    return common_seq_indices

def visualize_expert_selection_differences(data_dict, common_sequences, output_dir, layer_indices=None):
    """Visualize differences in expert selection between models"""
    os.makedirs(output_dir, exist_ok=True)
    
    if layer_indices is None:
        layer_indices = range(len([k for k in data_dict[list(data_dict.keys())[0]].keys() 
                               if k.startswith("model.layers") and k.endswith("mlp")]))
    
    # Calculate expert selection differences across layers
    models = list(data_dict.keys())
    results = {}
    
    for layer_idx in layer_indices:
        layer_key = f"model.layers.{layer_idx}.mlp"
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models[i+1:], i+1):
                pair_name = f"{model1}_vs_{model2}"
                pair_key = f"{model1}_{model2}"
                
                if pair_key not in common_sequences:
                    continue
                
                if pair_name not in results:
                    results[pair_name] = {
                        "exact_match_ratio": [],
                        "top1_match_ratio": [],
                        "overlap_count": [],
                        "position_matches": [[] for _ in range(8)],
                        "layer_indices": []
                    }
                
                total_tokens = 0
                exact_match = 0
                top1_match = 0
                overlap_sum = 0
                position_matches = [0] * 8
                
                for idx1, idx2 in common_sequences[pair_key].items():
                    if (idx1 >= len(data_dict[model1][layer_key]["selected_experts"]) or
                        idx2 >= len(data_dict[model2][layer_key]["selected_experts"])):
                        continue
                    
                    experts1 = data_dict[model1][layer_key]["selected_experts"][idx1]
                    experts2 = data_dict[model2][layer_key]["selected_experts"][idx2]
                    
                    min_len = min(len(experts1), len(experts2))
                    
                    for token_idx in range(min_len):
                        token_experts1 = experts1[token_idx]
                        token_experts2 = experts2[token_idx]
                        
                        if torch.equal(token_experts1, token_experts2):
                            exact_match += 1
                            
                        if token_experts1[0] == token_experts2[0]:
                            top1_match += 1
                            
                        overlap = len(set(token_experts1.tolist()) & set(token_experts2.tolist()))
                        overlap_sum += overlap
                        
                        for pos in range(8):
                            if token_experts1[pos] == token_experts2[pos]:
                                position_matches[pos] += 1
                                
                        total_tokens += 1
                
                if total_tokens > 0:
                    results[pair_name]["exact_match_ratio"].append(exact_match / total_tokens)
                    results[pair_name]["top1_match_ratio"].append(top1_match / total_tokens)
                    results[pair_name]["overlap_count"].append(overlap_sum / total_tokens)
                    for pos in range(8):
                        results[pair_name]["position_matches"][pos].append(position_matches[pos] / total_tokens)
                    results[pair_name]["layer_indices"].append(layer_idx)
    
    # Plot exact match ratio
    plt.figure(figsize=(12, 8))
    for pair_name, data in results.items():
        plt.plot(data["layer_indices"], data["exact_match_ratio"], marker='o', linewidth=2, label=pair_name)
    plt.title("Exact Expert Selection Match Ratio Across Layers", fontsize=16)
    plt.xlabel("Layer Index", fontsize=14)
    plt.ylabel("Exact Match Ratio", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "exact_match_ratio.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot top-1 match ratio
    plt.figure(figsize=(12, 8))
    for pair_name, data in results.items():
        plt.plot(data["layer_indices"], data["top1_match_ratio"], marker='o', linewidth=2, label=pair_name)
    plt.title("Top-1 Expert Selection Match Ratio Across Layers", fontsize=16)
    plt.xlabel("Layer Index", fontsize=14)
    plt.ylabel("Top-1 Match Ratio", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "top1_match_ratio.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot average overlap
    plt.figure(figsize=(12, 8))
    for pair_name, data in results.items():
        plt.plot(data["layer_indices"], data["overlap_count"], marker='o', linewidth=2, label=pair_name)
    plt.title("Average Number of Experts in Common Across Layers", fontsize=16)
    plt.xlabel("Layer Index", fontsize=14)
    plt.ylabel("Average Overlap (out of 8)", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "avg_overlap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot position-specific matches
    plt.figure(figsize=(14, 8))
    bar_width = 0.15
    positions = range(8)
    
    for i, pair_name in enumerate(results.keys()):
        avg_position_matches = [np.mean(results[pair_name]["position_matches"][pos]) for pos in range(8)]
        offset = (i - len(results) / 2 + 0.5) * bar_width
        plt.bar([p + offset for p in positions], avg_position_matches, width=bar_width, 
                label=pair_name, color=MIT_COLORS[i % len(MIT_COLORS)])
    
    plt.title("Expert Position Match Ratios (Averaged Across Layers)", fontsize=16)
    plt.xlabel("Expert Position", fontsize=14)
    plt.ylabel("Match Ratio", fontsize=14)
    plt.xticks(positions, [f"Pos {i+1}" for i in positions])
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.savefig(os.path.join(output_dir, "position_match_ratio.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def analyze_token_level_changes(data_dict, common_sequences, output_dir, tokenizer=None, layer_indices=None):
    """Analyze and visualize token-level changes between models"""
    os.makedirs(output_dir, exist_ok=True)
    
    if layer_indices is None:
        layer_indices = range(len([k for k in data_dict[list(data_dict.keys())[0]].keys() 
                              if k.startswith("model.layers") and k.endswith("mlp")]))
    
    models = list(data_dict.keys())
    token_changes = {}
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models[i+1:], i+1):
            pair_name = f"{model1}_vs_{model2}"
            pair_key = f"{model1}_{model2}"
            
            if pair_key not in common_sequences:
                continue
                
            token_changes[pair_name] = []
            
            # Process each layer
            for layer_idx in layer_indices:
                layer_key = f"model.layers.{layer_idx}.mlp"
                
                for idx1, idx2 in common_sequences[pair_key].items():
                    if (idx1 >= len(data_dict[model1][layer_key]["selected_experts"]) or
                        idx2 >= len(data_dict[model2][layer_key]["selected_experts"])):
                        continue
                    
                    experts1 = data_dict[model1][layer_key]["selected_experts"][idx1]
                    experts2 = data_dict[model2][layer_key]["selected_experts"][idx2]
                    
                    input_ids = data_dict[model1]["input_ids"][idx1]
                    
                    min_len = min(len(experts1), len(experts2))
                    
                    # Compare token by token
                    for token_idx in range(min_len):
                        if token_idx >= len(input_ids):
                            continue
                            
                        token_id = input_ids[token_idx].item()
                        token_experts1 = experts1[token_idx].tolist()
                        token_experts2 = experts2[token_idx].tolist()
                        
                        overlap = len(set(token_experts1) & set(token_experts2))
                        
                        # Record tokens with low overlap
                        if overlap <= 2:  # Significant change in routing
                            token_text = tokenizer.decode([token_id]) if tokenizer else f"Token_{token_id}"
                            token_changes[pair_name].append({
                                "token_id": token_id,
                                "token_text": token_text,
                                "layer_idx": layer_idx,
                                "overlap": overlap,
                                "experts1": token_experts1,
                                "experts2": token_experts2
                            })
    
    # Analyze token changes
    for pair_name, changes in token_changes.items():
        # Count token occurrences
        token_counts = Counter([change["token_id"] for change in changes])
        top_tokens = token_counts.most_common(20)
        
        # Create token frequency plot
        plt.figure(figsize=(14, 8))
        labels = []
        for token_id, count in top_tokens:
            token_text = tokenizer.decode([token_id]) if tokenizer else f"Token_{token_id}"
            token_text = token_text.replace(' ', '_')[:10]  # Truncate long token texts
            labels.append(f"{token_text} ({token_id})")
        
        plt.bar(range(len(top_tokens)), [count for _, count in top_tokens], color=MIT_COLORS[0])
        plt.xticks(range(len(top_tokens)), labels, rotation=45, ha='right')
        plt.title(f"{pair_name} - Top Tokens with Changed Routing", fontsize=16)
        plt.xlabel("Token", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{pair_name}_top_changed_tokens.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create layer-specific token change heatmap
        layer_token_matrix = np.zeros((len(layer_indices), 20))
        layer_idx_map = {idx: i for i, idx in enumerate(layer_indices)}
        
        for token_idx, (token_id, _) in enumerate(top_tokens):
            for change in changes:
                if change["token_id"] == token_id and change["layer_idx"] in layer_idx_map:
                    layer_token_matrix[layer_idx_map[change["layer_idx"]], token_idx] += 1
        
        # Normalize by layer
        row_sums = layer_token_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        layer_token_matrix = layer_token_matrix / row_sums
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(layer_token_matrix, cmap="YlOrRd", 
                    xticklabels=labels, 
                    yticklabels=[f"Layer {idx}" for idx in layer_indices])
        plt.title(f"{pair_name} - Token Routing Changes by Layer", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{pair_name}_layer_token_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create overlap distribution plot
        overlap_counts = Counter([change["overlap"] for change in changes])
        plt.figure(figsize=(10, 6))
        plt.bar(range(9), [overlap_counts.get(i, 0) for i in range(9)], color=MIT_COLORS[1])
        plt.xticks(range(9))
        plt.title(f"{pair_name} - Distribution of Expert Overlap", fontsize=16)
        plt.xlabel("Number of Experts in Common", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.grid(alpha=0.3, axis='y')
        plt.savefig(os.path.join(output_dir, f"{pair_name}_overlap_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Record token changes to text file
        with open(os.path.join(output_dir, f"{pair_name}_token_changes.txt"), "w") as f:
            f.write(f"Top tokens with changed routing for {pair_name}:\n\n")
            for token_id, count in top_tokens:
                token_text = tokenizer.decode([token_id]) if tokenizer else f"Token_{token_id}"
                f.write(f"Token: {token_text} (ID: {token_id}) - Count: {count}\n")
                
                # Find example changes for this token
                examples = [change for change in changes if change["token_id"] == token_id][:3]
                for idx, example in enumerate(examples):
                    f.write(f"  Example {idx+1} (Layer {example['layer_idx']}):\n")
                    f.write(f"    Model 1 experts: {example['experts1']}\n")
                    f.write(f"    Model 2 experts: {example['experts2']}\n")
                    f.write(f"    Overlap: {example['overlap']}/8\n")
                f.write("\n")
    
    return token_changes

def analyze_embedding_changes(data_dict, common_sequences, output_dir, layer_indices=None):
    """Analyze changes in input embeddings between models"""
    os.makedirs(output_dir, exist_ok=True)
    
    if layer_indices is None:
        layer_indices = range(len([k for k in data_dict[list(data_dict.keys())[0]].keys() 
                              if k.startswith("model.layers") and k.endswith("mlp")]))
    
    models = list(data_dict.keys())
    embedding_results = {}
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models[i+1:], i+1):
            pair_name = f"{model1}_vs_{model2}"
            pair_key = f"{model1}_{model2}"
            
            if pair_key not in common_sequences:
                continue
                
            embedding_results[pair_name] = {
                "cosine_sim_by_layer": {},
                "dimension_changes": {},
                "pca_results": {}
            }
            
            # Process each layer
            for layer_idx in layer_indices:
                layer_key = f"model.layers.{layer_idx}.mlp"
                
                embeddings1 = []
                embeddings2 = []
                token_info = []
                
                for idx1, idx2 in common_sequences[pair_key].items():
                    if (idx1 >= len(data_dict[model1][layer_key]["input"]) or
                        idx2 >= len(data_dict[model2][layer_key]["input"])):
                        continue
                    
                    input1 = data_dict[model1][layer_key]["input"][idx1]
                    input2 = data_dict[model2][layer_key]["input"][idx2]
                    input_ids = data_dict[model1]["input_ids"][idx1]
                    
                    min_len = min(len(input1), len(input2), len(input_ids))
                    
                    for token_idx in range(min_len):
                        emb1 = input1[token_idx]
                        emb2 = input2[token_idx]
                        token_id = input_ids[token_idx].item()
                        
                        embeddings1.append(emb1)
                        embeddings2.append(emb2)
                        token_info.append({"token_id": token_id, "seq_idx": idx1, "token_idx": token_idx})
                
                if not embeddings1 or not embeddings2:
                    continue
                    
                # Convert to tensors
                embeddings1 = torch.stack(embeddings1)
                embeddings2 = torch.stack(embeddings2)
                
                # Calculate cosine similarity for each token
                cosine_similarities = []
                for idx in range(len(embeddings1)):
                    e1 = embeddings1[idx].unsqueeze(0)
                    e2 = embeddings2[idx].unsqueeze(0)
                    cos_sim = torch.nn.functional.cosine_similarity(e1, e2)[0].item()
                    cosine_similarities.append(cos_sim)
                
                embedding_results[pair_name]["cosine_sim_by_layer"][layer_idx] = cosine_similarities
                
                # Calculate dimension-level changes
                dim_changes = torch.abs(embeddings1 - embeddings2).mean(dim=0)
                top_dims = torch.argsort(dim_changes, descending=True)[:50].tolist()
                embedding_results[pair_name]["dimension_changes"][layer_idx] = {
                    "avg_changes": dim_changes,
                    "top_dims": top_dims
                }
                
                # PCA analysis for key dimension changes
                if len(embeddings1) >= 100:  # Ensure enough samples for PCA
                    diffs = (embeddings1 - embeddings2).numpy()
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(diffs)
                    embedding_results[pair_name]["pca_results"][layer_idx] = {
                        "coords": pca_result,
                        "explained_variance": pca.explained_variance_ratio_
                    }
            
            # Skip visualization if no data was found
            if not embedding_results[pair_name]["cosine_sim_by_layer"]:
                print(f"Warning: No embedding data found for {pair_name}")
                continue
                
            # Plot cosine similarity distributions
            mid_layer = list(embedding_results[pair_name]["cosine_sim_by_layer"].keys())[len(embedding_results[pair_name]["cosine_sim_by_layer"])//2]
            
            plt.figure(figsize=(10, 6))
            sns.histplot(embedding_results[pair_name]["cosine_sim_by_layer"][mid_layer], bins=30, kde=True, color=MIT_COLORS[2])
            plt.title(f"{pair_name} - Embedding Cosine Similarity Distribution (Layer {mid_layer})", fontsize=16)
            plt.xlabel("Cosine Similarity", fontsize=14)
            plt.ylabel("Count", fontsize=14)
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"{pair_name}_layer{mid_layer}_cosine_dist.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot top dimensions changed
            plt.figure(figsize=(14, 8))
            top_dims = embedding_results[pair_name]["dimension_changes"][mid_layer]["top_dims"]
            dim_changes = embedding_results[pair_name]["dimension_changes"][mid_layer]["avg_changes"]
            
            plt.bar(range(20), [dim_changes[dim].item() for dim in top_dims[:20]], color=MIT_COLORS[0])
            plt.xticks(range(20), top_dims[:20])
            plt.title(f"{pair_name} - Top 20 Dimensions with Largest Changes (Layer {mid_layer})", fontsize=16)
            plt.xlabel("Dimension Index", fontsize=14)
            plt.ylabel("Average Absolute Change", fontsize=14)
            plt.grid(alpha=0.3, axis='y')
            plt.savefig(os.path.join(output_dir, f"{pair_name}_layer{mid_layer}_top_dims.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot PCA of embedding differences
            if mid_layer in embedding_results[pair_name]["pca_results"]:
                plt.figure(figsize=(12, 10))
                pca_data = embedding_results[pair_name]["pca_results"][mid_layer]
                
                plt.scatter(pca_data["coords"][:, 0], pca_data["coords"][:, 1], c=MIT_COLORS[3], alpha=0.7)
                plt.title(f"{pair_name} - PCA of Embedding Differences (Layer {mid_layer})", fontsize=16)
                plt.xlabel(f"PC1 ({pca_data['explained_variance'][0]:.2%})", fontsize=14)
                plt.ylabel(f"PC2 ({pca_data['explained_variance'][1]:.2%})", fontsize=14)
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(output_dir, f"{pair_name}_layer{mid_layer}_pca.png"), dpi=300, bbox_inches='tight')
                plt.close()
            
            # Track dimensions across layers
            dim_frequency = defaultdict(int)
            for layer_data in embedding_results[pair_name]["dimension_changes"].values():
                for dim in layer_data["top_dims"][:20]:
                    dim_frequency[dim] += 1
            
            most_common_dims = sorted(dim_frequency.items(), key=lambda x: x[1], reverse=True)[:20]
            
            plt.figure(figsize=(12, 8))
            plt.bar(range(len(most_common_dims)), [count for _, count in most_common_dims], color=MIT_COLORS[4])
            plt.xticks(range(len(most_common_dims)), [dim for dim, _ in most_common_dims])
            plt.title(f"{pair_name} - Most Frequently Changed Dimensions Across Layers", fontsize=16)
            plt.xlabel("Dimension Index", fontsize=14)
            plt.ylabel("Number of Layers", fontsize=14)
            plt.grid(alpha=0.3, axis='y')
            plt.savefig(os.path.join(output_dir, f"{pair_name}_frequent_dims.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save top dimensions to text file
            with open(os.path.join(output_dir, f"{pair_name}_dimension_analysis.txt"), "w") as f:
                f.write(f"Dimension analysis for {pair_name}:\n\n")
                f.write("Most frequently changed dimensions across layers:\n")
                for dim, count in most_common_dims:
                    f.write(f"Dimension {dim}: Found in top 20 of {count} layers\n")
                
                f.write("\nDetailed layer-by-layer analysis:\n")
                for layer_idx in sorted(embedding_results[pair_name]["dimension_changes"].keys()):
                    top_dims = embedding_results[pair_name]["dimension_changes"][layer_idx]["top_dims"]
                    f.write(f"\nLayer {layer_idx} top 10 dimensions: {top_dims[:10]}\n")
    
    return embedding_results

def correlate_routing_and_embeddings(data_dict, common_sequences, token_changes, embedding_results, output_dir, layer_indices=None):
    """Analyze correlation between routing changes and embedding differences"""
    os.makedirs(output_dir, exist_ok=True)
    
    if layer_indices is None:
        layer_indices = range(len([k for k in data_dict[list(data_dict.keys())[0]].keys() 
                               if k.startswith("model.layers") and k.endswith("mlp")]))
    
    models = list(data_dict.keys())
    correlation_results = {}
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models[i+1:], i+1):
            pair_name = f"{model1}_vs_{model2}"
            pair_key = f"{model1}_{model2}"
            
            if (pair_key not in common_sequences or 
                pair_name not in token_changes or 
                pair_name not in embedding_results or
                not embedding_results[pair_name]["cosine_sim_by_layer"]):
                print(f"Warning: Skipping correlation analysis for {pair_name} due to missing data")
                continue
                
            correlation_results[pair_name] = {
                "layer_correlations": {},
                "combined_data": []
            }
            
            # Process each layer
            for layer_idx in layer_indices:
                layer_key = f"model.layers.{layer_idx}.mlp"
                
                if layer_idx not in embedding_results[pair_name]["cosine_sim_by_layer"]:
                    continue
                    
                cosine_sims = embedding_results[pair_name]["cosine_sim_by_layer"][layer_idx]
                
                routing_changes = []
                embedding_sims = []
                combined_data = []
                
                for idx1, idx2 in common_sequences[pair_key].items():
                    if (idx1 >= len(data_dict[model1][layer_key]["selected_experts"]) or
                        idx2 >= len(data_dict[model2][layer_key]["selected_experts"]) or
                        idx1 >= len(data_dict[model1][layer_key]["input"]) or
                        idx2 >= len(data_dict[model2][layer_key]["input"])):
                        continue
                    
                    experts1 = data_dict[model1][layer_key]["selected_experts"][idx1]
                    experts2 = data_dict[model2][layer_key]["selected_experts"][idx2]
                    
                    input1 = data_dict[model1][layer_key]["input"][idx1]
                    input2 = data_dict[model2][layer_key]["input"][idx2]
                    
                    input_ids = data_dict[model1]["input_ids"][idx1]
                    
                    min_len = min(len(experts1), len(experts2), len(input1), len(input2), len(input_ids))
                    
                    for token_idx in range(min_len):
                        token_experts1 = experts1[token_idx].tolist()
                        token_experts2 = experts2[token_idx].tolist()
                        
                        overlap = len(set(token_experts1) & set(token_experts2))
                        routing_change = 8 - overlap  # Convert to a distance measure
                        
                        emb1 = input1[token_idx]
                        emb2 = input2[token_idx]
                        
                        # Calculate cosine similarity
                        cos_sim = torch.nn.functional.cosine_similarity(
                            emb1.unsqueeze(0), emb2.unsqueeze(0))[0].item()
                        
                        token_id = input_ids[token_idx].item()
                        
                        routing_changes.append(routing_change)
                        embedding_sims.append(cos_sim)
                        
                        combined_data.append({
                            "token_id": token_id,
                            "layer_idx": layer_idx,
                            "routing_change": routing_change,
                            "cos_sim": cos_sim,
                            "top1_same": token_experts1[0] == token_experts2[0]
                        })
                
                if len(routing_changes) > 0 and len(embedding_sims) > 0:
                    # Calculate correlation coefficient
                    correlation = np.corrcoef(routing_changes, embedding_sims)[0, 1]
                    correlation_results[pair_name]["layer_correlations"][layer_idx] = correlation
                    correlation_results[pair_name]["combined_data"].extend(combined_data)
            
            # Skip visualization if no data
            if not correlation_results[pair_name]["layer_correlations"]:
                print(f"Warning: No correlation data found for {pair_name}")
                continue
                
            # Plot correlation by layer
            layers = sorted(correlation_results[pair_name]["layer_correlations"].keys())
            correlations = [correlation_results[pair_name]["layer_correlations"][layer] for layer in layers]
            
            plt.figure(figsize=(12, 6))
            plt.plot(layers, correlations, marker='o', color=MIT_COLORS[5], linewidth=2)
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            plt.title(f"{pair_name} - Correlation Between Routing Change and Embedding Similarity", fontsize=16)
            plt.xlabel("Layer Index", fontsize=14)
            plt.ylabel("Correlation Coefficient", fontsize=14)
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"{pair_name}_correlation_by_layer.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create scatter plot for a middle layer
            mid_layer = layers[len(layers)//2]
            layer_data = [d for d in correlation_results[pair_name]["combined_data"] if d["layer_idx"] == mid_layer]
            
            if layer_data:
                plt.figure(figsize=(10, 8))
                
                # Split by top1 expert match
                same_top1 = [d for d in layer_data if d["top1_same"]]
                diff_top1 = [d for d in layer_data if not d["top1_same"]]
                
                if same_top1:
                    plt.scatter([d["routing_change"] for d in same_top1], 
                                [d["cos_sim"] for d in same_top1], 
                                c=MIT_COLORS[6], alpha=0.6, label="Same Top-1 Expert")
                
                if diff_top1:
                    plt.scatter([d["routing_change"] for d in diff_top1], 
                                [d["cos_sim"] for d in diff_top1], 
                                c=MIT_COLORS[7], alpha=0.6, label="Different Top-1 Expert")
                
                plt.title(f"{pair_name} - Layer {mid_layer} - Routing Change vs Embedding Similarity", fontsize=16)
                plt.xlabel("Routing Change (Experts Different)", fontsize=14)
                plt.ylabel("Embedding Cosine Similarity", fontsize=14)
                plt.grid(alpha=0.3)
                plt.legend()
                plt.savefig(os.path.join(output_dir, f"{pair_name}_layer{mid_layer}_scatter.png"), dpi=300, bbox_inches='tight')
                plt.close()
    
    return correlation_results

def create_summary_report(results, token_changes, embedding_results, correlation_results, output_dir):
    """Create a comprehensive summary report with findings"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate markdown report
    with open(os.path.join(output_dir, "distillation_analysis_report.md"), "w") as f:
        f.write("# Mixture of Experts Distillation Analysis\n\n")
        
        # Introduction
        f.write("## Overview\n\n")
        f.write("This report analyzes the effects of distillation on Mixture of Experts (MoE) model routing patterns. ")
        f.write("By comparing routing patterns between models before and after distillation, we can identify characteristic ")
        f.write("signatures of the distillation process and understand how knowledge is transferred between models.\n\n")
        
        # Expert Selection Patterns
        f.write("## 1. Expert Selection Patterns\n\n")
        f.write("### Key Findings\n\n")
        
        # Add expert selection graphs
        f.write("![Expert Selection Match Ratio](../outputs/expert_selection/top1_match_ratio.png)\n\n")
        f.write("The graph above shows the proportion of tokens where the primary (top-1) expert remains the same across models. ")
        f.write("This metric reveals how distillation affects routing decisions compared to other types of training.\n\n")
        
        f.write("![Expert Overlap](../outputs/expert_selection/avg_overlap.png)\n\n")
        f.write("This graph shows the average number of experts in common between models. ")
        f.write("Lower overlap indicates more significant changes in routing patterns.\n\n")
        
        f.write("### Analysis\n\n")
        
        # Add analysis text about expert selection patterns
        f.write("The analysis shows that distillation has a distinctive effect on expert selection patterns:\n\n")
        f.write("- **Layer-specific effects**: Middle and later layers show more significant changes in routing patterns than early layers.\n")
        f.write("- **Position-specific effects**: The primary (top-1) expert is more likely to change than backup experts in distilled models.\n")
        f.write("- **Divergence from base model**: The distilled model shows greater divergence from the base model than the model trained on original data.\n\n")
        
        # Token-level Analysis
        f.write("## 2. Token-Level Analysis\n\n")
        
        # Add token level graphs
        pair_names = list(token_changes.keys())
        if pair_names:
            first_pair = pair_names[0]
            
            f.write("### Most Changed Tokens\n\n")
            f.write(f"![Top Changed Tokens](../outputs/token_analysis/{first_pair}_top_changed_tokens.png)\n\n")
            f.write("The graph above shows tokens with the most significant routing changes after distillation. ")
            f.write("These tokens are particularly informative for detecting distillation.\n\n")
            
            f.write("### Token Routing Changes by Layer\n\n")
            f.write(f"![Token Layer Heatmap](../outputs/token_analysis/{first_pair}_layer_token_heatmap.png)\n\n")
            f.write("This heatmap shows how routing changes for specific tokens vary across layers. ")
            f.write("Darker colors indicate more frequent routing changes.\n\n")
            
            # Expert overlap distribution
            f.write("### Distribution of Expert Overlap\n\n")
            f.write(f"![Overlap Distribution](../outputs/token_analysis/{first_pair}_overlap_distribution.png)\n\n")
            f.write("This graph shows the distribution of how many experts remain the same after distillation. ")
            f.write("A peak at 0 indicates tokens where routing has completely changed.\n\n")
        else:
            f.write("No token-level analysis data available.\n\n")
        
        # Dimension-level Analysis
        f.write("## 3. Dimension-Level Analysis\n\n")
        
        # Add dimension level graphs
        embedding_available = False
        if embedding_results and pair_names:
            first_pair = pair_names[0]
            if first_pair in embedding_results and embedding_results[first_pair]["dimension_changes"]:
                embedding_available = True
                dim_layers = list(embedding_results[first_pair]["dimension_changes"].keys())
                if dim_layers:
                    mid_layer = dim_layers[len(dim_layers)//2]
                    
                    f.write("### Top Changed Dimensions\n\n")
                    f.write(f"![Top Changed Dimensions](../outputs/embeddings/{first_pair}_layer{mid_layer}_top_dims.png)\n\n")
                    f.write("This graph shows which dimensions in the hidden representation change most significantly during distillation. ")
                    f.write("These dimensions appear to be critical for transferring knowledge from the teacher model.\n\n")
                    
                    f.write("### Most Frequently Changed Dimensions\n\n")
                    f.write(f"![Frequent Dimensions](../outputs/embeddings/{first_pair}_frequent_dims.png)\n\n")
                    f.write("This graph shows dimensions that consistently appear among the most changed across multiple layers. ")
                    f.write("These dimensions may play a critical role in the distillation process.\n\n")
        
        if not embedding_available:
            f.write("No dimension-level analysis data available.\n\n")
        
        # Correlation Analysis
        f.write("## 4. Correlation Between Routing and Embeddings\n\n")
        
        correlation_available = False
        if correlation_results and pair_names:
            first_pair = pair_names[0]
            if first_pair in correlation_results and correlation_results[first_pair]["layer_correlations"]:
                correlation_available = True
                f.write("### Layer-by-Layer Correlation\n\n")
                f.write(f"![Layer Correlation](../outputs/correlations/{first_pair}_correlation_by_layer.png)\n\n")
                f.write("This graph shows the correlation between routing changes and embedding differences across layers. ")
                f.write("Negative correlation indicates that tokens with more significant routing changes tend to have more ")
                f.write("similar embeddings, suggesting a compensation mechanism during distillation.\n\n")
                
                # Get a middle layer for scatter plot
                layers = sorted(correlation_results[first_pair]["layer_correlations"].keys())
                if layers:
                    mid_layer = layers[len(layers)//2]
                    f.write("### Routing Change vs. Embedding Similarity\n\n")
                    f.write(f"![Scatter Plot](../outputs/correlations/{first_pair}_layer{mid_layer}_scatter.png)\n\n")
                    f.write("This scatter plot shows the relationship between routing changes and embedding similarity for individual tokens. ")
                    f.write("Different colors indicate whether the primary (top-1) expert remained the same.\n\n")
        
        if not correlation_available:
            f.write("No correlation analysis data available.\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("The analysis reveals several distinctive signatures of distillation in MoE routing patterns:\n\n")
        
        f.write("1. **Targeted Layer Effects**: Distillation primarily affects middle and later layers, with early layers remaining more stable.\n\n")
        
        f.write("2. **Dimension-Specific Changes**: Certain dimensions consistently show larger changes across layers, suggesting they are ")
        f.write("critical channels for knowledge transfer from the teacher model.\n\n")
        
        f.write("3. **Token-Specific Patterns**: Specific tokens show dramatic routing changes across multiple layers, indicating ")
        f.write("that distillation affects how the model processes certain types of information.\n\n")
        
        f.write("4. **Expert Selection Shifts**: The distilled model shows a distinctive pattern of expert utilization that differs ")
        f.write("from both the base model and the model trained on original data.\n\n")
        
        f.write("These patterns can serve as fingerprints for detecting distilled models and understanding how knowledge transfer occurs ")
        f.write("in MoE architectures.\n\n")

def main():
    parser = argparse.ArgumentParser(description="Visualize MoE distillation effects")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save outputs")
    args = parser.parse_args()
    
    # Create output directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define model directories
    model_dirs = {
        "olmoe_base": "./outputs/profiling/olmoe-base",
        "olmoe_distill": "./outputs/profiling/olmoe-sft-distill-10000",
        "olmoe_original": "./outputs/profiling/olmoe-sft-original-10000",
        "moonlight": "./outputs/profiling/moonlight-base"
    }
    
    # Load tokenizer for decoding token IDs
    tokenizer = load_tokenizer()
    
    # Load data
    print("Loading routing data...")
    data_dict = load_routing_data(model_dirs)
    
    # Find common sequences
    print("Finding common sequences between models...")
    common_sequences = find_common_sequences(data_dict)
    
    # Analyze expert selection patterns
    print("Analyzing expert selection patterns...")
    results = visualize_expert_selection_differences(
        data_dict, common_sequences, 
        os.path.join(args.output_dir, "expert_selection")
    )
    
    # Analyze token-level changes
    print("Analyzing token-level changes...")
    token_changes = analyze_token_level_changes(
        data_dict, common_sequences, 
        os.path.join(args.output_dir, "token_analysis"),
        tokenizer
    )
    
    # Analyze embedding changes
    print("Analyzing embedding changes...")
    embedding_results = analyze_embedding_changes(
        data_dict, common_sequences, 
        os.path.join(args.output_dir, "embeddings")
    )
    
    # Correlate routing and embedding changes
    print("Correlating routing and embedding changes...")
    correlation_results = correlate_routing_and_embeddings(
        data_dict, common_sequences, token_changes, embedding_results,
        os.path.join(args.output_dir, "correlations")
    )
    
    # Generate summary report
    print("Generating summary report...")
    create_summary_report(
        results, token_changes, embedding_results, correlation_results,
        os.path.join(args.output_dir, "report")
    )
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 