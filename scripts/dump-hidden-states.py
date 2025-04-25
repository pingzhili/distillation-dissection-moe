import os
from loguru import logger
from functools import partial

import torch
from accelerate.utils import set_seed
from datasets import load_dataset
from fire import Fire
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

from ddmoe.data import batch_preprocess_fn, CustomDataCollatorWithPadding
from ddmoe.models.deepseek import DeepseekV3ForCausalLM, DeepseekV3MoE

set_seed(233)


def dump_router_token_hidden_states(
    checkpoint_path: str, save_dir: str, valid_dataset: str,
):
    if "moonlight" in checkpoint_path.lower():
        model = DeepseekV3ForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True).cuda()
    if "olmoe" in checkpoint_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            "allenai/OLMoE-1B-7B-0125-Instruct", trust_remote_code=True
        )
    elif "moonlight" in checkpoint_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            "moonshotai/Moonlight-16B-A3B-Instruct", trust_remote_code=True
        )
    else:
        raise NotImplementedError(f"Tokenizer for {checkpoint_path} not implemented.")

    if valid_dataset is not None:
        logger.info(f"Loading valid dataset from {valid_dataset}")
        dataset = load_dataset("csv", data_files=valid_dataset)["train"]
    else:  
        logger.info(f"Loading default valid dataset from Phando/sft-dataset-valid")
        dataset = load_dataset("Phando/sft-dataset-valid", split="train", trust_remote_code=True)
    preprocess_fn = partial(batch_preprocess_fn, task="chat-profile", tokenizer=tokenizer)
    columns = dataset.column_names
    logger.info(f"Columns: {columns}")
    dataset = dataset.map(preprocess_fn, batched=True, num_proc=8, remove_columns=["question", "response"])
    logger.info(f"Dataset[0]: {dataset[0]}")
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=CustomDataCollatorWithPadding(tokenizer=tokenizer, extra_keys_to_ignore=["source"]),
        shuffle=True,
    )

    routing_hidden_states_per_module = {}


    # custom forward on MoE block
    def get_custom_olmoe_forward(module_name: str):
        def _custom_forward(self, hidden_states: torch.Tensor):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            assert batch_size == 1, "Batch size must be 1 for router token hidden states dumping"

            # Added
            routing_hidden_states_per_module[module_name]["input"].append(hidden_states.squeeze().detach().cpu())

            hidden_states = hidden_states.view(-1, hidden_dim)
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate(hidden_states)

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

            # Added
            routing_hidden_states_per_module[module_name]["selected_experts"].append(selected_experts.squeeze().detach().cpu())

            if self.norm_topk_prob:
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )

            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be selected
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

            # Loop over all available experts in the model and perform the computation on each expert
            for expert_idx in range(self.num_experts):
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx])

                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

            return final_hidden_states, router_logits

        return _custom_forward

    def get_custom_moonlight_forward(module_name: str):
        def _custom_forward(self, hidden_states: torch.Tensor):
            routing_hidden_states_per_module[module_name]["input"].append(hidden_states.squeeze().detach().cpu())
            identity = hidden_states
            orig_shape = hidden_states.shape
            topk_idx, topk_weight, aux_loss, logits = self.gate(hidden_states)
            routing_hidden_states_per_module[module_name]["selected_experts"].append(
                topk_idx.squeeze().detach().cpu())
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            flat_topk_idx = topk_idx.view(-1)
            if not self.training:
                y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
            if self.config.n_shared_experts is not None:
                y = y + self.shared_experts(identity)
            return y, logits

        return _custom_forward

    input_id_list = []
    sources_list = []
    for name, module in model.named_modules():
        if isinstance(module, OlmoeSparseMoeBlock):
            module.forward = get_custom_olmoe_forward(name).__get__(module, type(module))
            routing_hidden_states_per_module[name] = {"input": [], "selected_experts": []}
        elif isinstance(module, DeepseekV3MoE):
            module.forward = get_custom_moonlight_forward(name).__get__(module, type(module))
            routing_hidden_states_per_module[name] = {"input": [], "selected_experts": []}

    for batch in tqdm(data_loader, desc=f"Dumping router token hidden states..."):
        sources_list.append(batch.pop("source")[0])
        batch = {k: v.cuda() for k, v in batch.items()}
        # for k, v in batch.items():
        #     batch[k] = v.squeeze(0)
        input_id_list.append(batch["input_ids"].squeeze(0).detach().cpu())
        with torch.no_grad():
            _ = model(**batch)

    routing_hidden_states_per_module["input_ids"] = input_id_list
    routing_hidden_states_per_module["sources"] = sources_list
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(routing_hidden_states_per_module, os.path.join(save_dir, "router_tokens.pt"))


def main(content: str, checkpoint_path: str, save_dir: str, valid_dataset: str = None):
    logger.info(f"Dumping {content} from {checkpoint_path} to {save_dir}")
    if content == "router_tokens":
        dump_router_token_hidden_states(checkpoint_path, save_dir=save_dir, valid_dataset=valid_dataset)
    else:
        raise ValueError(f"Invalid content: {content}")


if __name__ == "__main__":
    Fire(main)
