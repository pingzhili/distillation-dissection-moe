import os
import random

import torch
from accelerate.utils import set_seed
from datasets import Dataset, load_dataset
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

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


def dump_hidden_states(checkpoint_path: str):
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True).cuda()
    if "olmoe" in checkpoint_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            "allenai/OLMoE-1B-7B-0125-Instruct", trust_remote_code=True
        )
    else:
        raise NotImplementedError(f"Tokenizer for {checkpoint_path} not implemented.")

    dataset = get_wikitext2(tokenizer=tokenizer, seqlen=512, nsamples=4, split="train")
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
            outputs = model(**batch, output_router_logits=True, output_hidden_states=True, output_attentions=True)
        outputs_list.append({
            "logits": outputs.logits.squeeze(),
            "hidden_states": outputs.hidden_states,
            "router_logits": outputs.router_logits,
            "attention": outputs.attentions,
        })

    save_dir = os.path.join(checkpoint_path, "profiling")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(outputs_list, os.path.join(save_dir, "outputs.pt"))


if __name__ == "__main__":
    Fire(dump_hidden_states)
