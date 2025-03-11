import os

import torch
from fire import Fire


def compare_olmoe_routing_results(
        before_router_checkpoint_path: str,
        after_router_checkpoint_path: str,
        save_dir: str = "./results",
):
    before_router_hidden_states = torch.load(before_router_checkpoint_path)
    after_router_hidden_states = torch.load(after_router_checkpoint_path)

    # Dictionary to store input_ids with different routing results
    different_routing_inputs = {}

    # Iterate through each sample
    # Keys are: dict_keys(['model.layers.0.mlp', 'model.layers.1.mlp', 'model.layers.2.mlp', 'model.layers.3.mlp', 'model.layers.4.mlp', 'model.layers.5.mlp', 'model.layers.6.mlp', 'model.layers.7.mlp', 'model.layers.8.mlp', 'model.layers.9.mlp', 'model.layers.10.mlp', 'model.layers.11.mlp', 'model.layers.12.mlp', 'model.layers.13.mlp', 'model.layers.14.mlp', 'model.layers.15.mlp', 'input_ids'])
    aligned_after_router_hidden_states = {}
    aligned_after_router_hidden_states["input_ids"] = []
    num_layers = len(before_router_hidden_states) - 1
    num_samples = len(before_router_hidden_states["input_ids"])
    for i in range(num_layers):
        aligned_after_router_hidden_states[f"model.layers.{i}.mlp"] = []

    for input_ids in before_router_hidden_states["input_ids"]:
        aligned_after_router_hidden_states["input_ids"].append(input_ids)
        original_after_idx = None
        for i in range(num_samples):
            if input_ids.shape[0] == after_router_hidden_states["input_ids"][i].shape[0] and torch.allclose(
                    input_ids, after_router_hidden_states["input_ids"][i]):
                original_after_idx = i
                break
        for i in range(num_layers):
            aligned_after_router_hidden_states[f"model.layers.{i}.mlp"].append(
                after_router_hidden_states[f"model.layers.{i}.mlp"][original_after_idx])

    # Compare the routing results and capture the tokens with different routing results
    different_routing_list = []  # list of (token input_id, layer_id, before_routing, after_routing, before_token_input, after_token_input)
    for sample_id in range(num_samples):
        for layer_id in range(num_layers):
            before_routing = before_router_hidden_states[f"model.layers.{layer_id}.mlp"]["selected_experts"][sample_id]
            after_routing = aligned_after_router_hidden_states[f"model.layers.{layer_id}.mlp"]["selected_experts"][
                sample_id]
            before_input = before_router_hidden_states[f"model.layers.{layer_id}.mlp"]["input"][sample_id]
            after_input = aligned_after_router_hidden_states[f"model.layers.{layer_id}.mlp"]["input"][sample_id]
            for token_id in range(before_routing.shape[0]):
                if not torch.allclose(before_routing[token_id], after_routing[token_id]):
                    different_routing_list.append((
                        before_router_hidden_states["input_ids"][sample_id][token_id].item(),
                        layer_id,
                        before_routing[token_id].item(),
                        after_routing[token_id].item(),
                        before_input[token_id],
                        after_input[token_id],
                    ))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(different_routing_list, os.path.join(save_dir, "different_routing_tokens.pt"))


if __name__ == "__main__":
    Fire(compare_olmoe_routing_results)
