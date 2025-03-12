import os
import random

import torch
from fire import Fire
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

random.seed(233)


def compare_olmoe_routing_results(
        before_router_checkpoint_path: str,
        after_router_checkpoint_path: str,
        save_dir: str = "./results",
):
    before_router_hidden_states = torch.load(before_router_checkpoint_path, map_location="cuda")
    print(f"Loaded before router hidden states from {before_router_checkpoint_path}")
    after_router_hidden_states = torch.load(after_router_checkpoint_path, map_location="cuda")
    print(f"Loaded after router hidden states from {after_router_checkpoint_path}")

    # Iterate through each sample
    # Keys are: dict_keys(['model.layers.0.mlp', 'model.layers.1.mlp', 'model.layers.2.mlp', 'model.layers.3.mlp', 'model.layers.4.mlp', 'model.layers.5.mlp', 'model.layers.6.mlp', 'model.layers.7.mlp', 'model.layers.8.mlp', 'model.layers.9.mlp', 'model.layers.10.mlp', 'model.layers.11.mlp', 'model.layers.12.mlp', 'model.layers.13.mlp', 'model.layers.14.mlp', 'model.layers.15.mlp', 'input_ids'])
    aligned_after_router_hidden_states = {}
    aligned_after_router_hidden_states["input_ids"] = []
    num_layers = len(before_router_hidden_states) - 1
    num_samples = len(before_router_hidden_states["input_ids"])
    for i in range(num_layers):
        aligned_after_router_hidden_states[f"model.layers.{i}.mlp"] = {}
        aligned_after_router_hidden_states[f"model.layers.{i}.mlp"]["selected_experts"] = []
        aligned_after_router_hidden_states[f"model.layers.{i}.mlp"]["input"] = []

    for input_ids in tqdm(before_router_hidden_states["input_ids"], desc="Aligning hidden states..."):
        aligned_after_router_hidden_states["input_ids"].append(input_ids)
        original_after_idx = None
        for i in range(num_samples):
            if input_ids.shape[0] == after_router_hidden_states["input_ids"][i].shape[0] and torch.allclose(
                    input_ids, after_router_hidden_states["input_ids"][i]):
                original_after_idx = i
                break
        for i in range(num_layers):
            aligned_after_router_hidden_states[f"model.layers.{i}.mlp"]["selected_experts"].append(
                after_router_hidden_states[f"model.layers.{i}.mlp"]["selected_experts"][original_after_idx]
            )
            aligned_after_router_hidden_states[f"model.layers.{i}.mlp"]["input"].append(
                after_router_hidden_states[f"model.layers.{i}.mlp"]["input"][original_after_idx]
            )

    # Compare the routing results and capture the tokens with different routing results
    different_routing_list = []  # list of (token input_id, layer_id, before_routing, after_routing, before_token_input, after_token_input)
    progress_bar = tqdm(total=num_samples * num_layers,
                        desc=f"Comparing routing results on {num_samples} samples x {num_layers} layers...")
    total_num_tokens = 0
    # number of different routing experts count
    num_experts_per_token = 8
    diff_experts_count = {i: 0 for i in range(num_experts_per_token + 1)}

    for sample_id in range(num_samples):
        for layer_id in range(num_layers):
            before_routing = before_router_hidden_states[f"model.layers.{layer_id}.mlp"]["selected_experts"][sample_id]
            after_routing = aligned_after_router_hidden_states[f"model.layers.{layer_id}.mlp"]["selected_experts"][
                sample_id]
            before_input = before_router_hidden_states[f"model.layers.{layer_id}.mlp"]["input"][sample_id]
            after_input = aligned_after_router_hidden_states[f"model.layers.{layer_id}.mlp"]["input"][sample_id]
            for token_id in range(before_routing.shape[0]):
                total_num_tokens += 1
                if not torch.allclose(before_routing[token_id], after_routing[token_id]):
                    different_routing_list.append((
                        before_router_hidden_states["input_ids"][sample_id][token_id].item(),
                        layer_id,
                        before_routing[token_id],
                        after_routing[token_id],
                        before_input[token_id],
                        after_input[token_id],
                    ))
                    num_diff_experts = torch.sum(before_routing[token_id] != after_routing[token_id]).item()
                    diff_experts_count[num_diff_experts] += 1
                else:
                    diff_experts_count[0] += 1

            progress_bar.update(1)

    progress_bar.close()
    print(f"Found {len(different_routing_list)} tokens with different routing results over {total_num_tokens} tokens")
    for i in range(num_experts_per_token + 1):
        print(f"# of tokens with {i} different experts: {diff_experts_count[i]}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, "different_routing_tokens.pt")
    print(f"Saving different routing tokens to {save_path}")
    torch.save(different_routing_list, save_path)
    print("Done!")


def calculate_expert_token_distibution(
        before_router_checkpoint_path: str,
        after_router_checkpoint_path: str,
        save_dir: str = "./results",
):
    # calculate the token distribution of each expert before and after the router, on a 2D plot using PCA
    before_router_hidden_states = torch.load(before_router_checkpoint_path, map_location="cuda")
    print(f"Loaded before router hidden states from {before_router_checkpoint_path}")
    after_router_hidden_states = torch.load(after_router_checkpoint_path, map_location="cuda")
    print(f"Loaded after router hidden states from {after_router_checkpoint_path}")

    num_layers = len(before_router_hidden_states) - 1
    num_samples = len(before_router_hidden_states["input_ids"])
    num_experts = before_router_hidden_states[f"model.layers.0.mlp"]["input"][0].shape[0]
    num_routed_experts_per_token = 8

    # Concatenate the input and selected experts for each layer, and then group them by expert
    before_expert_input_per_layer = {i: {j: [] for j in range(num_experts)} for i in range(num_layers)}
    after_expert_input_per_layer = {i: {j: [] for j in range(num_experts)} for i in range(num_layers)}

    progress_bar = tqdm(total=num_samples * num_layers, desc=f"Collecting expert input distributions...")

    interested_layers = list(range(0, num_layers, 4))
    for sample_id in range(num_samples):
        for layer_id in range(num_layers):
            if layer_id not in interested_layers:
                continue
            before_input = before_router_hidden_states[f"model.layers.{layer_id}.mlp"]["input"][sample_id]
            after_input = after_router_hidden_states[f"model.layers.{layer_id}.mlp"]["input"][sample_id]
            before_routing = before_router_hidden_states[f"model.layers.{layer_id}.mlp"]["selected_experts"][sample_id]
            after_routing = after_router_hidden_states[f"model.layers.{layer_id}.mlp"]["selected_experts"][sample_id]
            for token_id in range(before_routing.shape[0]):
                # 90% percent skip this token
                if random.random() < 0.8:
                    continue
                for i in range(num_routed_experts_per_token):
                    before_routed_expert = before_routing[token_id][i].item()
                    after_routed_expert = after_routing[token_id][i].item()
                    before_expert_input_per_layer[layer_id][before_routed_expert].append(before_input[token_id].cpu())
                    after_expert_input_per_layer[layer_id][after_routed_expert].append(after_input[token_id].cpu())

            progress_bar.update(1)

    progress_bar.close()

    # Visualize by randomly sampling a subset of tokens
    num_samples_per_expert = 128
    progress_bar = tqdm(total=num_layers * num_samples_per_expert, desc="Collecting expert input distributions...")
    for layer_id in range(num_layers):
        for expert_id in range(num_experts):
            before_expert_input = before_expert_input_per_layer[layer_id][expert_id]
            after_expert_input = after_expert_input_per_layer[layer_id][expert_id]
            if len(before_expert_input) == 0:
                continue
            before_sampled_input = random.sample(before_expert_input,
                                                 min(num_samples_per_expert, len(before_expert_input)))
            after_sampled_input = random.sample(after_expert_input,
                                                min(num_samples_per_expert, len(after_expert_input)))
            before_expert_input_per_layer[layer_id][expert_id] = torch.stack(before_sampled_input, dim=0)
            after_expert_input_per_layer[layer_id][expert_id] = torch.stack(after_sampled_input, dim=0)

            progress_bar.update(1)

    progress_bar.close()

    # Save the expert input distributions
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"expert_input_distributions_{num_samples_per_expert}.pt")
    print(f"Saving expert input distributions (randomly sampled {num_samples_per_expert}) to {save_path}")

    torch.save({
        "before_expert_input_per_layer": before_expert_input_per_layer,
        "after_expert_input_per_layer": after_expert_input_per_layer,
    }, save_path)

    # Visualize with MIT colors
    # PCA
    for layer_id in range(num_layers):
        print(f"Visualizing expert input distribution for layer {layer_id}...")
        before_expert_input = torch.cat([v for v in before_expert_input_per_layer[layer_id].values()], dim=0)
        after_expert_input = torch.cat([v for v in after_expert_input_per_layer[layer_id].values()], dim=0)
        pca = PCA(n_components=2)
        before_pca = pca.fit_transform(before_expert_input)
        after_pca = pca.fit_transform(after_expert_input)
        plt.figure(figsize=(8, 8))
        plt.scatter(before_pca[:, 0], before_pca[:, 1], c="blue", label="Before Router")
        plt.scatter(after_pca[:, 0], after_pca[:, 1], c="red", label="After Router")
        plt.title(f"Layer {layer_id} Expert Input Distribution")
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"layer_{layer_id}_expert_input_distribution.png"))
        plt.close()

    print("Done!")


if __name__ == "__main__":
    Fire(calculate_expert_token_distibution)
