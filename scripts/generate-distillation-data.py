from fire import Fire
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    set_seed, AutoTokenizer, DataCollatorWithPadding,
)
from ddmoe.data import batch_preprocess_fn
from ddmoe.models import DeepseekV3ForCausalLM
from functools import partial
from tqdm import tqdm
import os

set_seed(233)


def generate_distillation_data(
        model_name: str = "moonshotai/Moonlight-16B-A3B-Instruct",
        dataset_name: str = "ServiceNow-AI/R1-Distill-SFT",
        max_length: int = 1024,
        save_dir: str = "data/",
        num_workers: int = 4,
):
    model = DeepseekV3ForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = model.generation_config.eos_token_id[0]
    if dataset_name == "ServiceNow-AI/R1-Distill-SFT":
        dataset = load_dataset(
            "ServiceNow-AI/R1-Distill-SFT", "v1", trust_remote_code=True
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not found.")
    preprocess_fn = partial(batch_preprocess_fn, task="chat-eval", tokenizer=tokenizer)
    dataset = dataset.map(preprocess_fn, batched=True, num_proc=num_workers)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    dataloader = DataLoader(dataset=dataset, collate_fn=data_collator, batch_size=1, num_workers=num_workers)

    # write the response into a file on-the-fly
    with open(os.path.join(save_dir, "distillation_data.txt"), "w") as f:
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"]
            response = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
            response = tokenizer.batch_decode(response, skip_special_tokens=True)
            f.write(response[0] + "\n")


model = DeepseekV3ForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
