import json
import os
import time
from functools import partial

import openai
from datasets import load_dataset
from fire import Fire
from tqdm import tqdm
from transformers import set_seed

from ddmoe.data import batch_preprocess_fn

set_seed(233)


def append_generation(response, prompt, output_file):
    entry = {
        "response": response,
        "prompt": prompt,
    }
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def api_generate_distillation_data(
        dataset_name: str = "ServiceNow-AI/R1-Distill-SFT",
        base_url: str = "http://0.0.0.0:8014/v1",
        save_dir: str = "data/phimoe/",
        model_name: str = "microsoft/Phi-3.5-MoE-instruct",
        num_workers: int = 4,
):
    client = openai.Client(base_url=base_url, api_key="EMPTY")
    dataset = load_dataset(
        dataset_name, "v1", trust_remote_code=True
    )
    dataset = dataset["train"]
    # remove those samples with "source" is "ai2-adapt-dev/tulu_hard_coded_repeated_10"
    dataset = dataset.filter(lambda example: example["source"] != "ai2-adapt-dev/tulu_hard_coded_repeated_10")
    preprocess_fn = partial(batch_preprocess_fn, task="chat-gen")
    dataset = dataset.map(preprocess_fn, batched=True, num_proc=num_workers, remove_columns=dataset.column_names)
    batch_size = 128
    progress_bar = tqdm(
        total=len(dataset) // batch_size, desc=f"Generating distillation data via API (batch size is {batch_size})"
    )
    for i in range(0, len(dataset), batch_size):
        # write batch file on-the-fly into "_tmp_batch_input.jsonl"
        # format is like:
        # {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
        # {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
        batch = dataset[i:i + batch_size]["content"]  # of a list of messages
        with open("_tmp_batch_input.jsonl", 'w', encoding='utf-8') as f:
            for j, messages in enumerate(batch):
                request = {
                    "custom_id": f"request-{i * batch_size + j}",
                    "body": {
                        "model": model_name,
                        "url": "/v1/chat/completions",
                        "messages": messages,
                        "max_tokens": 1024
                    }
                }
                f.write(json.dumps(request, ensure_ascii=False) + "\n")
        with open("_tmp_batch_input.jsonl", "rb") as file:
            uploaded_file = client.files.create(file=file, purpose="batch")
        # call the API
        batch_job = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        # Monitor the batch job status
        while batch_job.status not in ["completed", "failed", "cancelled"]:
            time.sleep(3)  # Wait for 3 seconds before checking the status again
            batch_job = client.batches.retrieve(batch_job.id)
        if batch_job.status == "failed":
            print(f"Batch job failed with status: {batch_job.status}")
            print(f"Batch job errors: {batch_job.errors}")
            raise RuntimeError(f"Batch job failed at batch {i}")

        if batch_job.status == "completed":
            result_file_id = batch_job.output_file_id
            file_response = client.files.content(result_file_id)
            result_content = file_response.read()
            with open(os.path.join(save_dir, f"distillation_data.jsonl"), 'ab') as file:
                file.write(result_content)
            # also write messages in this batch along with custom_id into a separate file
            with open(os.path.join(save_dir, f"distillation_data_input.jsonl"), 'a', encoding='utf-8') as f:
                for j, messages in enumerate(batch):
                    entry = {
                        "custom_id": f"request-{i * batch_size + j}",
                        "messages": messages
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        else:
            print(f"Batch job failed with status: {batch_job.status}")
            return None
        progress_bar.update(1)


if __name__ == "__main__":
    Fire(api_generate_distillation_data)
