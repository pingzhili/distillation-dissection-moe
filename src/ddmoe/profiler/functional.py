import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F


def analyze_routing_entropy(router_logits_list, model_names, save_dir: str):
    # Calculate entropy of routing probabilities for each model
    results = {}
    for model_idx, model_name in enumerate(model_names):
        router_logits = router_logits_list[model_idx]

        # Get layer-wise entropies
        layer_entropies = []
        for layer_idx in range(len(router_logits)):
            # Convert logits to probabilities with softmax
            probs = F.softmax(router_logits[layer_idx], dim=-1)

            # Calculate entropy: -sum(p_i * log(p_i))
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            layer_entropies.append(entropy.mean().item())

        results[model_name] = layer_entropies

    # Visualization
    plt.figure(figsize=(12, 6))
    for model_name, entropies in results.items():
        plt.plot(range(len(entropies)), entropies, label=model_name, marker='o')

    plt.xlabel('Layer Index')
    plt.ylabel('Average Routing Entropy')
    plt.title('Routing Probability Entropy by Layer')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'routing_entropy_comparison.png'), dpi=300)
    torch.save(results, os.path.join(save_dir, 'routing_entropy_comparison.pth'))

    return results


def analyze_load_balancing(router_logits_list, model_names, save_dir: str):
    results = {}
    for model_idx, model_name in enumerate(model_names):
        router_logits = router_logits_list[model_idx]

        # Calculate expert utilization per layer
        layer_utilizations = []
        for layer_idx in range(len(router_logits)):
            logits = router_logits[layer_idx]

            # Get top-k routing decisions (top-8 for OLMoE, top-6+2 for MoonLight)
            k = 8  # Adjust as needed for the model
            if "Moonlight" in model_name:
                # Consider the shared experts mechanism if needed
                pass

            # Get the top-k expert indices for each token
            _, top_k_indices = torch.topk(logits, k=k, dim=-1)

            # Count how many times each expert is activated
            num_experts = logits.size(-1)  # Usually 64
            expert_counts = torch.zeros(num_experts)

            for token_idx in range(top_k_indices.size(0)):
                for expert_idx in top_k_indices[token_idx]:
                    expert_counts[expert_idx] += 1

            # Normalize to get utilization percentage
            expert_utilization = expert_counts / expert_counts.sum()

            # Calculate load balancing metrics
            layer_utilizations.append(expert_utilization.numpy())

        results[model_name] = layer_utilizations

    # Visualization - Example for a specific layer
    layer_to_visualize = 5  # Choose a representative layer

    plt.figure(figsize=(15, 5))
    for i, model_name in enumerate(model_names):
        plt.subplot(1, len(model_names), i + 1)
        sns.barplot(x=list(range(64)), y=results[model_name][layer_to_visualize])
        plt.title(f'{model_name} - Layer {layer_to_visualize} Expert Utilization')
        plt.xlabel('Expert ID')
        plt.ylabel('Utilization Ratio')
        plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'load_balancing_comparison.png'), dpi=300)
    torch.save(results, os.path.join(save_dir, 'load_balancing_comparison.pth'))

    return results


def analyze_routing_sparsity(router_logits_list, model_names, save_dir: str):
    results = {}
    for model_idx, model_name in enumerate(model_names):
        router_logits = router_logits_list[model_idx]

        # Analyze routing sparsity
        layer_sparsity = []
        max_logit_values = []

        for layer_idx in range(len(router_logits)):
            logits = router_logits[layer_idx]

            # Calculate max logit value for each token
            max_logit, _ = torch.max(logits, dim=-1)
            max_logit_values.append(max_logit.numpy())

            # Calculate sparsity metrics
            # 1. Get difference between max and mean logit
            mean_logit = torch.mean(logits, dim=-1)
            sparsity_score = max_logit - mean_logit
            layer_sparsity.append(sparsity_score.mean().item())

        results[model_name] = {
            "sparsity_scores": layer_sparsity,
            "max_logits": max_logit_values
        }

    # Visualization 1: Sparsity scores by layer
    plt.figure(figsize=(12, 6))
    for model_name, data in results.items():
        plt.plot(range(len(data["sparsity_scores"])), data["sparsity_scores"],
                 label=model_name, marker='o')

    plt.xlabel('Layer Index')
    plt.ylabel('Routing Sparsity Score')
    plt.title('Routing Sparsity by Layer')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'routing_sparsity_comparison.png'), dpi=300)
    torch.save(results, os.path.join(save_dir, 'routing_sparsity_comparison.pth'))

    # Visualization 2: Distribution of max logit values
    plt.figure(figsize=(15, 10))
    for i, model_name in enumerate(model_names):
        # Choose a few representative layers
        layers_to_show = [0, len(results[model_name]["max_logits"]) // 2, len(results[model_name]["max_logits"]) - 1]

        for j, layer_idx in enumerate(layers_to_show):
            plt.subplot(len(model_names), len(layers_to_show), i * len(layers_to_show) + j + 1)
            sns.histplot(results[model_name]["max_logits"][layer_idx], bins=50, kde=True)
            plt.title(f'{model_name} - Layer {layer_idx} Max Logits')
            plt.xlabel('Max Logit Value')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'max_logit_distributions.png'), dpi=300)

    return results


def analyze_expert_collaboration(router_logits_list, model_names, save_dir: str):
    results = {}
    for model_idx, model_name in enumerate(model_names):
        router_logits = router_logits_list[model_idx]

        # Analyze expert collaboration patterns
        layer_collaborations = []

        for layer_idx in range(len(router_logits)):
            logits = router_logits[layer_idx]

            # Get top-k routing decisions
            k = 8  # top-8 for OLMoE
            if "Moonlight" in model_name:
                k = 6  # Adjust for MoonLight's top-6 + 2 shared

            # Get the top-k expert indices for each token
            _, top_k_indices = torch.topk(logits, k=k, dim=-1)

            # Calculate co-activation matrix
            num_experts = logits.size(-1)  # Usually 64
            coactivation_matrix = torch.zeros(num_experts, num_experts)

            for token_idx in range(top_k_indices.size(0)):
                # For each pair of experts activated for this token
                for i in range(k):
                    expert_i = top_k_indices[token_idx, i].item()
                    for j in range(i + 1, k):
                        expert_j = top_k_indices[token_idx, j].item()
                        coactivation_matrix[expert_i, expert_j] += 1
                        coactivation_matrix[expert_j, expert_i] += 1

            # Normalize by total co-activations
            if coactivation_matrix.sum() > 0:
                coactivation_matrix = coactivation_matrix / coactivation_matrix.sum()

            layer_collaborations.append(coactivation_matrix.numpy())

        results[model_name] = layer_collaborations

    # Visualization - Heatmap of expert collaboration for a selected layer
    layer_to_visualize = 5

    plt.figure(figsize=(15, 15))
    for i, model_name in enumerate(model_names):
        plt.subplot(1, len(model_names), i + 1)
        sns.heatmap(results[model_name][layer_to_visualize],
                    cmap='viridis', vmin=0, vmax=results[model_name][layer_to_visualize].max(),
                    xticklabels=5, yticklabels=5)
        plt.title(f'{model_name} - Layer {layer_to_visualize} Expert Co-activation')
        plt.xlabel('Expert ID')
        plt.ylabel('Expert ID')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'expert_collaboration_comparison.png'), dpi=300)
    torch.save(results, os.path.join(save_dir, 'expert_collaboration_comparison.pth'))

    # Calculate collaboration pattern similarity between models
    similarity_matrix = np.zeros((len(model_names), len(model_names)))

    for i, model_i in enumerate(model_names):
        for j, model_j in enumerate(model_names):
            if i == j:
                similarity_matrix[i, j] = 1.0
                continue

            # Calculate cosine similarity between collaboration patterns
            layer_similarities = []
            for layer_idx in range(len(results[model_i])):
                pattern_i = results[model_i][layer_idx].flatten()
                pattern_j = results[model_j][layer_idx].flatten()

                # Cosine similarity
                similarity = np.dot(pattern_i, pattern_j) / (
                        np.linalg.norm(pattern_i) * np.linalg.norm(pattern_j) + 1e-8)
                layer_similarities.append(similarity)

            similarity_matrix[i, j] = np.mean(layer_similarities)

    # Visualize similarity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=model_names, yticklabels=model_names)
    plt.title('Collaboration Pattern Similarity Between Models')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'collaboration_similarity.png'), dpi=300)
    torch.save(similarity_matrix, os.path.join(save_dir, 'collaboration_similarity.pth'))

    return results, similarity_matrix
