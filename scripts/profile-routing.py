import os
import random

import torch
from accelerate.utils import set_seed
from datasets import Dataset, load_dataset
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

from ddmoe.models import DeepseekV3ForCausalLM
from ddmoe.profiler.functional import (
    analyze_expert_collaboration, analyze_load_balancing, analyze_routing_entropy, analyze_routing_sparsity,
)

set_seed(233)


def get_wikitext2(tokenizer, seqlen: int, nsamples: int, split: str = "train"):
    if split == "train":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif split == "validation":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        raise ValueError(f"Invalid split: {split}")
    # length of 288059 should be enough
    text = "".join([" \n" if s == "" else s for s in data["text"][:1000]])

    enc = tokenizer(text, return_tensors="pt")
    dataset = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return dataset


def get_routing_logits(checkpoint_path: str):
    if "moonlight" in checkpoint_path.lower():
        model = DeepseekV3ForCausalLM.from_pretrained(
            checkpoint_path, trust_remote_code=True, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, trust_remote_code=True, device_map="auto"
        )

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

    dataset = get_wikitext2(tokenizer=tokenizer, seqlen=512, nsamples=16, split="train")
    data_loader = DataLoader(
        Dataset.from_list(dataset),
        batch_size=1,
        collate_fn=default_data_collator,
        shuffle=True,
    )

    outputs_list = []

    for batch in tqdm(data_loader, desc=f"Dumping hidden states..."):
        batch = {k: v.cuda() for k, v in batch.items()}
        for k, v in batch.items():
            batch[k] = v.squeeze(0)
        with torch.no_grad():
            outputs = model(**batch, output_router_logits=True, output_hidden_states=False, output_attentions=False)
        outputs_list.append({
            "logits": outputs.logits.squeeze(),
            "hidden_states": outputs.hidden_states,
            "router_logits": outputs.router_logits,
            "attention": outputs.attentions,
        })

    # concate all routing logits for each layer
    all_routing_logits = [outputs['router_logits'] for outputs in outputs_list]  # List[Set[torch.Tensor]]
    all_routing_logits = [torch.stack(logits, dim=0).cpu() for logits in
                          all_routing_logits]  # List[torch.Tensor of shape (n_layers, seq_length, n_experts)]
    all_routing_logits = torch.concat(all_routing_logits, dim=1)  # shape (n_layers, n_samples * seq_length, n_experts)

    return all_routing_logits


def profile_hidden_states(
        all_checkpoint_paths: str,
        save_dir="./results",
):
    checkpoint_list = all_checkpoint_paths.split(",")
    model_names = [os.path.basename(checkpoint_path) for checkpoint_path in checkpoint_list]
    routing_logits_list = [get_routing_logits(checkpoint_path) for checkpoint_path in checkpoint_list]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    analyze_load_balancing(routing_logits_list, model_names, save_dir)
    analyze_routing_entropy(routing_logits_list, model_names, save_dir)
    analyze_routing_sparsity(routing_logits_list, model_names, save_dir)
    analyze_expert_collaboration(routing_logits_list, model_names, save_dir)


if __name__ == "__main__":
    Fire(profile_hidden_states)
