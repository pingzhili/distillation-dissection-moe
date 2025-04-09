import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import argparse
from collections import defaultdict

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
    common_sequences = {}
    
    # Extract input_ids from the first model
    first_model = list(data_dict.keys())[0]
    first_model_inputs = data_dict[first_model]["input_ids"]
    
    # For each input sequence in the first model
    for i, input_seq in enumerate(first_model_inputs):
        common_found = True
        # Check if this sequence exists in all other models
        for model_name in list(data_dict.keys())[1:]:
            found = False
            for j, other_seq in enumerate(data_dict[model_name]["input_ids"]):
                if torch.equal(input_seq, other_seq):
                    if model_name not in common_sequences:
                        common_sequences[model_name] = {}
                    common_sequences[model_name][i] = j
                    found = True
                    break
            if not found:
                common_found = False
                break
        
        if common_found:
            if first_model not in common_sequences:
                common_sequences[first_model] = {}
            common_sequences[first_model][i] = i
            
    return common_sequences

def analyze_token_embeddings(data_dict, common_sequences, layer_indices=None, token_limit=1000):
    """Analyze token embeddings for significant changes between models"""
    if layer_indices is None:
        # Use all layers
        layer_indices = range(len([k for k in data_dict[list(data_dict.keys())[0]].keys() 
                                if k.startswith("model.layers") and k.endswith("mlp")]))
    
    models = list(data_dict.keys())
    results = {}
    token_samples = []
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models[i+1:], i+1):
            pair_name = f"{model1}_vs_{model2}"
            pair_key = f"{model1}_{model2}"
            
            if pair_key not in common_sequences:
                continue
                
            results[pair_name] = {
                "layer_token_embeddings": {},
                "cosine_sim_per_token": [],
                "embedding_change_norms": [],
                "token_metadata": []
            }
            
            # Sample tokens to analyze (limit to token_limit for memory constraints)
            local_token_samples = []
            
            # Collect token samples
            for idx1, idx2 in common_sequences[pair_key].items():
                if idx1 >= len(data_dict[model1]["input_ids"]):
                    continue
                    
                input_ids = data_dict[model1]["input_ids"][idx1]
                
                # Sample tokens from this sequence
                seq_length = len(input_ids)
                sample_indices = np.linspace(0, seq_length-1, min(20, seq_length), dtype=int)
                
                for token_idx in sample_indices:
                    local_token_samples.append((idx1, idx2, token_idx))
                    
                    if len(local_token_samples) >= token_limit:
                        break
                        
                if len(local_token_samples) >= token_limit:
                    break
            
            token_samples.extend(local_token_samples)
            
            # Analyze selected token samples
            for layer_idx in layer_indices:
                layer_key = f"model.layers.{layer_idx}.mlp"
                
                embeddings1 = []
                embeddings2 = []
                
                for seq_idx1, seq_idx2, token_idx in local_token_samples:
                    # Skip if out of range
                    if (seq_idx1 >= len(data_dict[model1][layer_key]["input"]) or 
                        seq_idx2 >= len(data_dict[model2][layer_key]["input"])):
                        continue
                        
                    # Get token embeddings for this sequence/token
                    if (token_idx < len(data_dict[model1][layer_key]["input"][seq_idx1]) and
                        token_idx < len(data_dict[model2][layer_key]["input"][seq_idx2])):
                        
                        emb1 = data_dict[model1][layer_key]["input"][seq_idx1][token_idx]
                        emb2 = data_dict[model2][layer_key]["input"][seq_idx2][token_idx]
                        
                        embeddings1.append(emb1)
                        embeddings2.append(emb2)
                
                # Convert to tensors for easier computation
                if embeddings1 and embeddings2:
                    embeddings1 = torch.stack(embeddings1)
                    embeddings2 = torch.stack(embeddings2)
                    
                    # Calculate cosine similarity for each token
                    for idx in range(len(embeddings1)):
                        e1 = embeddings1[idx].unsqueeze(0)
                        e2 = embeddings2[idx].unsqueeze(0)
                        cos_sim = torch.nn.functional.cosine_similarity(e1, e2)[0].item()
                        
                        # Calculate L2 norm of the difference
                        emb_diff = e1 - e2
                        diff_norm = torch.norm(emb_diff).item()
                        
                        # Store results
                        results[pair_name]["cosine_sim_per_token"].append({
                            "layer": layer_idx,
                            "token_idx": local_token_samples[idx][2],
                            "seq_idx": local_token_samples[idx][0],
                            "cosine_sim": cos_sim,
                            "diff_norm": diff_norm
                        })
                    
                    # Store embeddings for this layer
                    results[pair_name]["layer_token_embeddings"][layer_idx] = {
                        "model1": embeddings1,
                        "model2": embeddings2
                    }
    
    return results, token_samples

def analyze_dimension_changes(embedding_results, output_dir):
    """Analyze which dimensions changed the most between models"""
    os.makedirs(output_dir, exist_ok=True)
    
    dimension_changes = {}
    
    for pair_name, results in embedding_results.items():
        dimension_changes[pair_name] = {}
        
        for layer_idx, embeddings in results["layer_token_embeddings"].items():
            emb1 = embeddings["model1"]
            emb2 = embeddings["model2"]
            
            # Calculate absolute changes per dimension
            diff = (emb1 - emb2).abs()
            avg_diff_per_dim = diff.mean(dim=0)
            
            # Calculate relative changes per dimension
            # To avoid division by zero, add a small epsilon
            relative_diff = diff / (emb1.abs() + 1e-8)
            avg_rel_diff_per_dim = relative_diff.mean(dim=0)
            
            dimension_changes[pair_name][layer_idx] = {
                "abs_diff": avg_diff_per_dim,
                "rel_diff": avg_rel_diff_per_dim
            }
        
        # Plot top changed dimensions for selected layers
        selected_layers = np.linspace(0, max(results["layer_token_embeddings"].keys()), 4, dtype=int)
        
        for layer_idx in selected_layers:
            if layer_idx not in dimension_changes[pair_name]:
                continue
                
            # Plot top absolute changes
            plt.figure(figsize=(14, 6))
            abs_diff = dimension_changes[pair_name][layer_idx]["abs_diff"]
            
            # Get top 50 changed dimensions
            top_dims = torch.argsort(abs_diff, descending=True)[:50]
            
            plt.bar(range(len(top_dims)), abs_diff[top_dims], color=MIT_COLORS[0])
            plt.title(f"{pair_name} - Layer {layer_idx} - Top 50 Dimensions with Largest Absolute Changes", fontsize=14)
            plt.xlabel("Dimension Index", fontsize=12)
            plt.ylabel("Average Absolute Change", fontsize=12)
            plt.xticks(range(len(top_dims)), top_dims.tolist(), rotation=90, fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{pair_name}_layer{layer_idx}_top_abs_dims.png"), dpi=300)
            plt.close()
            
            # Plot top relative changes
            plt.figure(figsize=(14, 6))
            rel_diff = dimension_changes[pair_name][layer_idx]["rel_diff"]
            
            # Get top 50 relatively changed dimensions
            top_rel_dims = torch.argsort(rel_diff, descending=True)[:50]
            
            plt.bar(range(len(top_rel_dims)), rel_diff[top_rel_dims], color=MIT_COLORS[1])
            plt.title(f"{pair_name} - Layer {layer_idx} - Top 50 Dimensions with Largest Relative Changes", fontsize=14)
            plt.xlabel("Dimension Index", fontsize=12)
            plt.ylabel("Average Relative Change", fontsize=12)
            plt.xticks(range(len(top_rel_dims)), top_rel_dims.tolist(), rotation=90, fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{pair_name}_layer{layer_idx}_top_rel_dims.png"), dpi=300)
            plt.close()
        
        # Compute global statistics across all layers
        all_abs_diffs = []
        all_layers = list(dimension_changes[pair_name].keys())
        for layer_idx in all_layers:
            all_abs_diffs.append(dimension_changes[pair_name][layer_idx]["abs_diff"])
        
        if all_abs_diffs:
            global_avg_diff = torch.stack(all_abs_diffs).mean(dim=0)
            
            # Plot global top changed dimensions
            plt.figure(figsize=(14, 6))
            top_global_dims = torch.argsort(global_avg_diff, descending=True)[:50]
            
            plt.bar(range(len(top_global_dims)), global_avg_diff[top_global_dims], color=MIT_COLORS[2])
            plt.title(f"{pair_name} - Top 50 Dimensions with Largest Absolute Changes (Across All Layers)", fontsize=14)
            plt.xlabel("Dimension Index", fontsize=12)
            plt.ylabel("Average Absolute Change", fontsize=12)
            plt.xticks(range(len(top_global_dims)), top_global_dims.tolist(), rotation=90, fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{pair_name}_global_top_dims.png"), dpi=300)
            plt.close()
    
    return dimension_changes

def visualize_token_embedding_changes(embedding_results, token_samples, data_dict, common_sequences, output_dir):
    """Visualize token embedding changes using dimensionality reduction"""
    os.makedirs(output_dir, exist_ok=True)
    
    for pair_name, results in embedding_results.items():
        models = pair_name.split("_vs_")
        model1, model2 = models[0], models[1]
        
        # Find tokens with largest changes
        token_changes = sorted(results["cosine_sim_per_token"], key=lambda x: x["cosine_sim"])[:100]
        
        # Select middle layer for visualization
        layers = list(results["layer_token_embeddings"].keys())
        if not layers:
            continue
        
        mid_layer = layers[len(layers) // 2]
        
        # Get embeddings for this layer
        emb1 = results["layer_token_embeddings"][mid_layer]["model1"]
        emb2 = results["layer_token_embeddings"][mid_layer]["model2"]
        
        # Run dimensionality reduction
        combined_embeddings = torch.cat([emb1, emb2], dim=0).numpy()
        
        # PCA for initial dimensionality reduction
        pca = PCA(n_components=50)
        pca_result = pca.fit_transform(combined_embeddings)
        
        # UMAP for visualization
        reducer = umap.UMAP(random_state=42, min_dist=0.1, n_neighbors=15)
        embedding = reducer.fit_transform(pca_result)
        
        # Split back into model1 and model2 embeddings
        n_samples = emb1.shape[0]
        emb1_2d = embedding[:n_samples]
        emb2_2d = embedding[n_samples:]
        
        # Create a scatter plot
        plt.figure(figsize=(12, 10))
        
        # Plot all points
        plt.scatter(emb1_2d[:, 0], emb1_2d[:, 1], c=MIT_COLORS[0], alpha=0.7, label=model1)
        plt.scatter(emb2_2d[:, 0], emb2_2d[:, 1], c=MIT_COLORS[1], alpha=0.7, label=model2)
        
        # Highlight pairs of tokens with largest changes
        for i, token_info in enumerate(token_changes[:20]):
            if i >= min(len(emb1_2d), len(emb2_2d)):
                break
                
            idx = i  # Use index of the sorted tokens
            plt.plot([emb1_2d[idx, 0], emb2_2d[idx, 0]], [emb1_2d[idx, 1], emb2_2d[idx, 1]], 
                     'k-', alpha=0.5, linewidth=0.5)
        
        plt.title(f"{pair_name} - Layer {mid_layer} Token Embeddings", fontsize=16)
        plt.xlabel("UMAP Dimension 1", fontsize=14)
        plt.ylabel("UMAP Dimension 2", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{pair_name}_layer{mid_layer}_embeddings.png"), dpi=300)
        plt.close()
        
        # Plot cosine similarity distribution
        plt.figure(figsize=(10, 6))
        
        cosine_sims = [item["cosine_sim"] for item in results["cosine_sim_per_token"] 
                      if item["layer"] == mid_layer]
        
        sns.histplot(cosine_sims, bins=30, kde=True, color=MIT_COLORS[3])
        plt.axvline(np.mean(cosine_sims), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(cosine_sims):.4f}')
        
        plt.title(f"{pair_name} - Layer {mid_layer} Token Embedding Cosine Similarity", fontsize=16)
        plt.xlabel("Cosine Similarity", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{pair_name}_layer{mid_layer}_cosine_dist.png"), dpi=300)
        plt.close()

def correlate_routing_with_embeddings(data_dict, common_sequences, embedding_results, token_samples, output_dir):
    """Analyze correlation between embedding changes and routing changes"""
    os.makedirs(output_dir, exist_ok=True)
    
    models = list(data_dict.keys())
    results = {}
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models[i+1:], i+1):
            pair_name = f"{model1}_vs_{model2}"
            pair_key = f"{model1}_{model2}"
            
            if pair_name not in embedding_results or pair_key not in common_sequences:
                continue
                
            results[pair_name] = {
                "corr_by_layer": {},
                "token_level_data": []
            }
            
            # Get list of layers
            layers = list(sorted([int(k.split('.')[2]) 
                               for k in data_dict[model1].keys() 
                               if k.startswith("model.layers") and k.endswith("mlp")]))
            
            # Analyze correlation for each layer
            for layer_idx in layers:
                layer_key = f"model.layers.{layer_idx}.mlp"
                
                if layer_idx not in embedding_results[pair_name]["layer_token_embeddings"]:
                    continue
                
                routing_changes = []
                embedding_diffs = []
                
                # Process each token sample that belongs to this model pair
                relevant_token_samples = [
                    (seq_idx1, seq_idx2, token_idx) for seq_idx1, seq_idx2, token_idx in token_samples
                    if seq_idx1 in common_sequences[pair_key]
                ]
                
                # Process each token sample
                for idx, (seq_idx1, seq_idx2, token_idx) in enumerate(relevant_token_samples):
                    if idx >= len(embedding_results[pair_name]["layer_token_embeddings"][layer_idx]["model1"]):
                        continue
                        
                    # Skip if out of range
                    if (seq_idx1 >= len(data_dict[model1][layer_key]["selected_experts"]) or 
                        seq_idx2 >= len(data_dict[model2][layer_key]["selected_experts"])):
                        continue
                    
                    # Get routing data
                    if (token_idx < len(data_dict[model1][layer_key]["selected_experts"][seq_idx1]) and
                        token_idx < len(data_dict[model2][layer_key]["selected_experts"][seq_idx2])):
                        
                        routing1 = data_dict[model1][layer_key]["selected_experts"][seq_idx1][token_idx]
                        routing2 = data_dict[model2][layer_key]["selected_experts"][seq_idx2][token_idx]
                        
                        # Calculate routing change metrics
                        routing_overlap = len(set(routing1.tolist()) & set(routing2.tolist()))
                        top1_same = (routing1[0] == routing2[0])
                        
                        # Get embedding diff
                        emb1 = embedding_results[pair_name]["layer_token_embeddings"][layer_idx]["model1"][idx]
                        emb2 = embedding_results[pair_name]["layer_token_embeddings"][layer_idx]["model2"][idx]
                        
                        emb_diff = torch.norm((emb1 - emb2)).item()
                        cos_sim = torch.nn.functional.cosine_similarity(
                            emb1.unsqueeze(0), emb2.unsqueeze(0))[0].item()
                        
                        # Record data
                        routing_changes.append(8 - routing_overlap)  # Convert to a distance measure
                        embedding_diffs.append(emb_diff)
                        
                        results[pair_name]["token_level_data"].append({
                            "layer": layer_idx,
                            "seq_idx": seq_idx1,
                            "token_idx": token_idx,
                            "routing_overlap": routing_overlap,
                            "top1_same": top1_same,
                            "emb_diff": emb_diff,
                            "cos_sim": cos_sim
                        })
                
                # Calculate correlation
                if routing_changes and embedding_diffs:
                    corr = np.corrcoef(routing_changes, embedding_diffs)[0, 1]
                    results[pair_name]["corr_by_layer"][layer_idx] = corr
            
            # Plot correlation across layers
            plt.figure(figsize=(10, 6))
            
            layers = sorted(results[pair_name]["corr_by_layer"].keys())
            corrs = [results[pair_name]["corr_by_layer"][l] for l in layers]
            
            plt.plot(layers, corrs, 'o-', linewidth=2, color=MIT_COLORS[4])
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            
            plt.title(f"{pair_name} - Correlation Between Routing Change and Embedding Difference", fontsize=16)
            plt.xlabel("Layer", fontsize=14)
            plt.ylabel("Correlation Coefficient", fontsize=14)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{pair_name}_routing_emb_correlation.png"), dpi=300)
            plt.close()
            
            # Create a scatter plot for middle layer
            mid_layer = layers[len(layers) // 2] if layers else None
            if mid_layer is not None:
                plt.figure(figsize=(10, 8))
                
                # Filter data for this layer
                layer_data = [item for item in results[pair_name]["token_level_data"] 
                             if item["layer"] == mid_layer]
                
                x = [item["routing_overlap"] for item in layer_data]
                y = [item["emb_diff"] for item in layer_data]
                c = [item["top1_same"] for item in layer_data]
                
                # Create scatter plot
                plt.scatter(x, y, c=['green' if val else 'red' for val in c], alpha=0.7)
                
                plt.title(f"{pair_name} - Layer {mid_layer} - Routing Overlap vs Embedding Difference", fontsize=16)
                plt.xlabel("Number of Shared Experts", fontsize=14)
                plt.ylabel("Embedding L2 Difference", fontsize=14)
                plt.colorbar(plt.cm.ScalarMappable(cmap='RdYlGn'), 
                            ticks=[0.25, 0.75], 
                            label="Top-1 Expert Match")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{pair_name}_layer{mid_layer}_routing_vs_emb.png"), dpi=300)
                plt.close()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze token embeddings and routing in MoE models")
    parser.add_argument("--output_dir", type=str, default="./outputs/token_analysis",
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
    
    # Analyze token embeddings
    print("Analyzing token embeddings...")
    embedding_results, token_samples = analyze_token_embeddings(data_dict, common_sequences)
    
    # Analyze dimension changes
    print("Analyzing dimension-level changes...")
    dimension_changes = analyze_dimension_changes(embedding_results, os.path.join(args.output_dir, "dimensions"))
    
    # Visualize token embedding changes
    print("Visualizing token embedding changes...")
    visualize_token_embedding_changes(embedding_results, token_samples, data_dict, common_sequences, 
                                     os.path.join(args.output_dir, "embeddings"))
    
    # Correlate routing with embeddings
    print("Correlating routing changes with embedding changes...")
    correlation_results = correlate_routing_with_embeddings(data_dict, common_sequences, 
                                                          embedding_results, token_samples,
                                                          os.path.join(args.output_dir, "correlations"))
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 