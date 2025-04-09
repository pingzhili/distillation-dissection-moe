import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pathlib import Path

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

def extract_embedding_data(data_dict, common_sequences, num_samples=10, max_tokens=100):
    """Extract embedding data for a sample of sequences and tokens"""
    models = list(data_dict.keys())
    layer_keys = [k for k in data_dict[models[0]].keys() 
                 if k.startswith("model.layers") and k.endswith("mlp")]
    
    embedding_data = {}
    
    # Process each model pair
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models[i+1:], i+1):
            pair_name = f"{model1}_vs_{model2}"
            pair_key = f"{model1}_{model2}"
            
            if pair_key not in common_sequences:
                continue
                
            embedding_data[pair_name] = {
                "sequences": [],
                "layer_data": {layer_idx: {"model1": [], "model2": [], "token_info": []} 
                              for layer_idx in range(len(layer_keys))}
            }
            
            # Sample some sequences
            seq_indices = list(common_sequences[pair_key].keys())
            if num_samples < len(seq_indices):
                seq_indices = np.random.choice(seq_indices, num_samples, replace=False)
            
            # Process each sampled sequence
            for seq_idx1 in seq_indices:
                seq_idx2 = common_sequences[pair_key][seq_idx1]
                
                embedding_data[pair_name]["sequences"].append((seq_idx1, seq_idx2))
                
                # Process each layer
                for layer_idx, layer_key in enumerate(layer_keys):
                    if (seq_idx1 >= len(data_dict[model1][layer_key]["input"]) or 
                        seq_idx2 >= len(data_dict[model2][layer_key]["input"])):
                        continue
                    
                    # Get input embeddings for this sequence
                    emb1 = data_dict[model1][layer_key]["input"][seq_idx1]
                    emb2 = data_dict[model2][layer_key]["input"][seq_idx2]
                    
                    # Sample tokens (limit to max_tokens for memory)
                    num_tokens = min(len(emb1), len(emb2))
                    token_indices = np.linspace(0, num_tokens-1, min(max_tokens, num_tokens), dtype=int)
                    
                    for token_idx in token_indices:
                        # Skip if out of range
                        if token_idx >= len(emb1) or token_idx >= len(emb2):
                            continue
                            
                        # Get embeddings for this token
                        token_emb1 = emb1[token_idx]
                        token_emb2 = emb2[token_idx]
                        
                        # Store embeddings and token info
                        embedding_data[pair_name]["layer_data"][layer_idx]["model1"].append(token_emb1)
                        embedding_data[pair_name]["layer_data"][layer_idx]["model2"].append(token_emb2)
                        embedding_data[pair_name]["layer_data"][layer_idx]["token_info"].append({
                            "seq_idx": seq_idx1,
                            "token_idx": token_idx
                        })
    
    return embedding_data

def analyze_dimension_changes(embedding_data, output_dir):
    """Analyze which dimensions change the most between models"""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for pair_name, pair_data in embedding_data.items():
        results[pair_name] = {
            "avg_change_per_dim": [],
            "avg_relative_change_per_dim": [],
            "top_dimensions": []
        }
        
        num_layers = len(pair_data["layer_data"])
        
        # Create a figure for the dimension heatmap
        plt.figure(figsize=(20, 10))
        
        # Process each layer
        for layer_idx in range(num_layers):
            layer_data = pair_data["layer_data"][layer_idx]
            
            if not layer_data["model1"] or not layer_data["model2"]:
                continue
            
            # Convert to tensors
            model1_embs = torch.stack(layer_data["model1"])
            model2_embs = torch.stack(layer_data["model2"])
            
            # Calculate changes per dimension
            changes = torch.abs(model1_embs - model2_embs)
            avg_change = torch.mean(changes, dim=0)
            
            # Calculate relative changes per dimension
            relative_changes = changes / (torch.abs(model1_embs) + 1e-8)
            avg_relative_change = torch.mean(relative_changes, dim=0)
            
            # Store results
            results[pair_name]["avg_change_per_dim"].append(avg_change)
            results[pair_name]["avg_relative_change_per_dim"].append(avg_relative_change)
            
            # Find top changed dimensions
            top_dims = torch.argsort(avg_change, descending=True)[:20].tolist()
            results[pair_name]["top_dimensions"].append(top_dims)
            
            # Add to heatmap
            plt.subplot(1, num_layers, layer_idx + 1)
            sns.heatmap(changes.T, cmap="viridis", xticklabels=False)
            plt.title(f"Layer {layer_idx} Dimension Changes")
            plt.ylabel("Dimension Index")
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{pair_name}_dimension_changes_heatmap.png"), dpi=300)
        plt.close()
        
        # Create a figure for the top dimensions per layer
        plt.figure(figsize=(15, 8))
        
        # Collect top dimensions across all layers
        all_top_dims = {}
        for layer_idx in range(num_layers):
            if layer_idx >= len(results[pair_name]["top_dimensions"]):
                continue
                
            for pos, dim_idx in enumerate(results[pair_name]["top_dimensions"][layer_idx]):
                if dim_idx not in all_top_dims:
                    all_top_dims[dim_idx] = {"count": 0, "avg_rank": 0, "layers": []}
                
                all_top_dims[dim_idx]["count"] += 1
                all_top_dims[dim_idx]["avg_rank"] += pos
                all_top_dims[dim_idx]["layers"].append(layer_idx)
        
        # Calculate average rank
        for dim_idx in all_top_dims:
            all_top_dims[dim_idx]["avg_rank"] /= all_top_dims[dim_idx]["count"]
        
        # Sort by count, then by average rank
        top_dims_overall = sorted(
            all_top_dims.items(), 
            key=lambda x: (-x[1]["count"], x[1]["avg_rank"])
        )[:20]
        
        # Plot
        plt.bar(
            range(len(top_dims_overall)), 
            [d[1]["count"] for d in top_dims_overall],
            color=MIT_COLORS[0]
        )
        plt.xticks(
            range(len(top_dims_overall)), 
            [d[0] for d in top_dims_overall],
            rotation=90
        )
        plt.title(f"{pair_name} - Top Dimensions Across Layers", fontsize=16)
        plt.xlabel("Dimension Index", fontsize=14)
        plt.ylabel("Number of Layers", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{pair_name}_top_dimensions_overall.png"), dpi=300)
        plt.close()
        
        # Create a figure for dimension changes across layers
        plt.figure(figsize=(15, 8))
        
        # Get top 5 dimensions overall
        top5_dims = [d[0] for d in top_dims_overall[:5]]
        
        # Plot changes for these dimensions across layers
        for dim_idx in top5_dims:
            dim_changes = [results[pair_name]["avg_change_per_dim"][l][dim_idx].item() 
                          for l in range(len(results[pair_name]["avg_change_per_dim"]))]
            plt.plot(range(len(dim_changes)), dim_changes, 
                     marker='o', linewidth=2, label=f"Dim {dim_idx}")
        
        plt.title(f"{pair_name} - Top 5 Dimensions Change Across Layers", fontsize=16)
        plt.xlabel("Layer Index", fontsize=14)
        plt.ylabel("Average Absolute Change", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{pair_name}_top_dims_across_layers.png"), dpi=300)
        plt.close()
    
    return results

def analyze_dimension_clustering(embedding_data, output_dir):
    """Analyze how dimensions cluster together based on their changes"""
    os.makedirs(output_dir, exist_ok=True)
    
    for pair_name, pair_data in embedding_data.items():
        num_layers = len(pair_data["layer_data"])
        
        # Process middle layer for visualization
        mid_layer = num_layers // 2
        
        layer_data = pair_data["layer_data"][mid_layer]
        
        if not layer_data["model1"] or not layer_data["model2"]:
            continue
        
        # Convert to tensors
        model1_embs = torch.stack(layer_data["model1"])
        model2_embs = torch.stack(layer_data["model2"])
        
        # Calculate changes per dimension
        changes = torch.abs(model1_embs - model2_embs)
        
        # Apply PCA to analyze the relationship between dimensions
        # Transpose to have dimensions as samples
        changes_t = changes.T.numpy()
        
        if changes_t.shape[0] > 2:  # Need at least 2 dimensions for PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(changes_t)
            
            # Calculate correlation between dimensions
            corr_matrix = np.corrcoef(changes_t)
            
            # Plot PCA result
            plt.figure(figsize=(12, 10))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
            
            # Annotate top dimensions (those with highest average change)
            avg_changes = torch.mean(changes, dim=0).numpy()
            top_dims = np.argsort(avg_changes)[-20:]
            
            for dim_idx in top_dims:
                plt.annotate(
                    str(dim_idx),
                    (pca_result[dim_idx, 0], pca_result[dim_idx, 1]),
                    fontsize=9,
                    alpha=0.8
                )
            
            plt.title(f"{pair_name} - Layer {mid_layer} - Dimension Clustering (PCA)", fontsize=16)
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2f}%)", fontsize=14)
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2f}%)", fontsize=14)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{pair_name}_layer{mid_layer}_dim_pca.png"), dpi=300)
            plt.close()
            
            # Plot correlation heatmap (focused on top dimensions)
            plt.figure(figsize=(14, 12))
            sns.heatmap(
                corr_matrix[np.ix_(top_dims, top_dims)],
                cmap="coolwarm",
                annot=True,
                fmt=".2f",
                xticklabels=top_dims,
                yticklabels=top_dims
            )
            plt.title(f"{pair_name} - Layer {mid_layer} - Dimension Correlation (Top 20)", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{pair_name}_layer{mid_layer}_dim_corr.png"), dpi=300)
            plt.close()
            
            # Create a graph of highly correlated dimensions
            G = nx.Graph()
            
            # Add nodes
            for dim_idx in top_dims:
                G.add_node(dim_idx, size=avg_changes[dim_idx] * 500)
            
            # Add edges for highly correlated pairs
            threshold = 0.6  # Correlation threshold
            for i in range(len(top_dims)):
                for j in range(i+1, len(top_dims)):
                    dim1 = top_dims[i]
                    dim2 = top_dims[j]
                    corr = corr_matrix[dim1, dim2]
                    if abs(corr) > threshold:
                        G.add_edge(dim1, dim2, weight=abs(corr), color='red' if corr < 0 else 'blue')
            
            # Draw the graph
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes
            node_sizes = [G.nodes[n]["size"] for n in G.nodes]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=MIT_COLORS[0], alpha=0.8)
            
            # Draw edges with colors based on correlation sign
            edges = G.edges()
            colors = [G[u][v]["color"] for u, v in edges]
            weights = [G[u][v]["weight"] * 3 for u, v in edges]
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color=colors, alpha=0.6)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10)
            
            plt.title(f"{pair_name} - Layer {mid_layer} - Dimension Correlation Network", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{pair_name}_layer{mid_layer}_dim_network.png"), dpi=300)
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze dimension-level changes in MoE models")
    parser.add_argument("--output_dir", type=str, default="./outputs/dimension_analysis",
                        help="Directory to save analysis results")
    args = parser.parse_args()
    
    # Define model directories
    model_dirs = {
        "olmoe_base": "./outputs/profiling/olmoe-base",
        "olmoe_distill": "./outputs/profiling/olmoe-sft-distill-10000",
        "olmoe_original": "./outputs/profiling/olmoe-sft-original-10000",
        "moonlight": "./outputs/profiling/moonlight-base"
    }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load routing data
    data_dict = load_routing_data(model_dirs)
    
    # Find common sequences
    print("Finding common sequences between models...")
    common_sequences = find_common_sequences(data_dict)
    
    # Extract embedding data
    print("Extracting embedding data for analysis...")
    embedding_data = extract_embedding_data(data_dict, common_sequences)
    
    # Analyze dimension changes
    print("Analyzing dimension changes...")
    dimension_results = analyze_dimension_changes(embedding_data, 
                                               os.path.join(args.output_dir, "dimension_changes"))
    
    # Analyze dimension clustering
    print("Analyzing dimension clustering...")
    analyze_dimension_clustering(embedding_data, 
                               os.path.join(args.output_dir, "dimension_clustering"))
    
    # Save summary of top dimensions per model pair
    summary_file = os.path.join(args.output_dir, "top_dimensions_summary.txt")
    with open(summary_file, "w") as f:
        for pair_name, results in dimension_results.items():
            f.write(f"Top dimensions for {pair_name}:\n")
            
            # Collect all top dimensions across layers
            all_top_dims = {}
            for layer_idx, top_dims in enumerate(results["top_dimensions"]):
                f.write(f"  Layer {layer_idx}: {top_dims[:10]}\n")
                
                for pos, dim_idx in enumerate(top_dims):
                    if dim_idx not in all_top_dims:
                        all_top_dims[dim_idx] = {"count": 0, "layers": []}
                    
                    all_top_dims[dim_idx]["count"] += 1
                    all_top_dims[dim_idx]["layers"].append(layer_idx)
            
            # Sort by count
            top_dims_overall = sorted(
                all_top_dims.items(), 
                key=lambda x: -x[1]["count"]
            )[:20]
            
            f.write("\n  Top dimensions across all layers:\n")
            for dim_idx, info in top_dims_overall:
                f.write(f"    Dimension {dim_idx}: Appeared in {info['count']} layers {info['layers']}\n")
            
            f.write("\n")
    
    print(f"Analysis complete. Results saved to {args.output_dir}")
    print(f"Top dimensions summary saved to {summary_file}")

if __name__ == "__main__":
    main() 