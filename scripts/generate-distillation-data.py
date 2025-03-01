import torch
from fire import Fire
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import set_seed, AutoTokenizer
from ddmoe.data import batch_preprocess_fn, CustomDataCollatorWithPadding
from ddmoe.models import DeepseekV3ForCausalLM
import deepspeed
import openai
from functools import partial
from tqdm import tqdm
import os
import json

set_seed(233)


def append_generation(response, prompt, output_file):
    entry = {
        "response": response,
        "prompt": prompt,
    }
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_generate_distillation_data(
        model_name: str = "moonshotai/Moonlight-16B-A3B-Instruct",
        dataset_name: str = "ServiceNow-AI/R1-Distill-SFT",
        max_length: int = 1024,
        save_dir: str = "data/",
        num_workers: int = 4,
):
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    model = DeepseekV3ForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", trust_remote_code=True, device_map=f"cuda:{local_rank}"
    )
    model = deepspeed.init_inference(
        model, tensor_parallel=2,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = model.generation_config.eos_token_id[0]
    if dataset_name == "ServiceNow-AI/R1-Distill-SFT":
        dataset = load_dataset(
            dataset_name, "v1", trust_remote_code=True
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not found.")
    dataset = dataset["train"]
    preprocess_fn = partial(batch_preprocess_fn, task="chat-eval", tokenizer=tokenizer)
    dataset = dataset.map(preprocess_fn, batched=True, num_proc=num_workers, remove_columns=dataset.column_names)
    data_collator = CustomDataCollatorWithPadding(
        tokenizer=tokenizer, pad_to_multiple_of=8, extra_keys_to_ignore=["content"]
    )
    dataloader = DataLoader(dataset=dataset, collate_fn=data_collator, batch_size=1, num_workers=num_workers)

    # write the response into a file on-the-fly
    for batch in tqdm(dataloader, desc="Generating distillation data"):
        input_ids = batch["input_ids"].cuda()
        content = batch["content"]
        with torch.inference_mode():
            generated_ids = model.generate(inputs=input_ids, max_new_tokens=max_length - len(input_ids[0]))
        generated_ids = generated_ids[:, len(input_ids[0]):]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        append_generation(response, content[0], os.path.join(save_dir, f"distillation_data_rank_{rank}.jsonl"))

def api_generate_distillation_data(
        dataset_name: str = "ServiceNow-AI/R1-Distill-SFT",
        base_url: str="http://127.0.0.1:30000/v1",
        save_dir: str = "data/",
        num_workers: int = 4,
):
    client = openai.Client(base_url=base_url, api_key="EMPTY")
    dataset = load_dataset(
        dataset_name, "v1", trust_remote_code=True
    )
    dataset = dataset["train"]
    preprocess_fn = partial(batch_preprocess_fn, task="chat-eval")
    dataset = dataset.map(preprocess_fn, batched=True, num_proc=num_workers, remove_columns=dataset.column_names)
    for i, messages in enumerate(tqdm(dataset, desc="Generating distillation data via API")):
        print(messages)
        response = client.chat.completions.create(
            model="default",
            messages=messages,
        )
        print(f">> {i} :", response)
        print()


if __name__ == "__main__":
    Fire(api_generate_distillation_data)
