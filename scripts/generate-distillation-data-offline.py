import json
import os
from functools import partial

import shutil
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams
from datasets import load_dataset
from fire import Fire
from loguru import logger
from tqdm import tqdm
from transformers import set_seed, AutoModelForCausalLM

from ddmoe.data import batch_preprocess_fn

set_seed(233)


def append_generation(response, prompt, output_file):
    entry = {
        "response": response,
        "prompt": prompt,
    }
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def api_generate_distillation_data_eager(
        dataset_name: str = "openai/gsm8k",
        save_dir: str = "data/r1-qwen-7b-gsm8k/",
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        is_eval: bool = False,
        num_workers: int = 4,
        num_samples: int = None,
        num_gpus: int = 1,
        num_splits: int = None,
        split_id: int = 0,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logger.info(f"Created directory {save_dir}")

    if "qwen3" in model_name.lower():
        tokenizer = "Qwen/Qwen3-8B"
    elif "qwen" in model_name.lower() or "qwen3" in model_name.lower():
        tokenizer = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    elif "llama-3.2" in model_name.lower():
        tokenizer = "meta-llama/Llama-3.2-1B-Instruct"
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise NotImplementedError(f"Model {model_name} is not supported")
    
    if "lm_head.pt" in model_name:
        model_dir = os.path.dirname(model_name)
        if "qwen3-8b" in model_name.lower():
            tmp_model = AutoModelForCausalLM.from_pretrained(tokenizer, torch_dtype=torch.bfloat16)
            # copy the lm_head_weight to the tmp_model
            lm_head_weight = torch.load(model_name)["teacher_model.lm_head.weight"]
            # save weight to tmp_model
            tmp_model.lm_head.weight.data.copy_(lm_head_weight)
            logger.info(f"Copied lm_head_weight and saving full model to {os.path.join(model_dir, 'tmp_model')}")
            tmp_model.save_pretrained(os.path.join(model_dir, "tmp_model"))
            model_name = os.path.join(model_dir, "tmp_model")
            logger.info(f"Using tmp_model {model_name} for generation")
        else:
            raise NotImplementedError(f"Model {model_name} is confusing")

    
    is_gsm_8k = "gsm8k" in dataset_name.lower()
    sampling_params = SamplingParams(max_tokens=4096)
    llm = LLM(
        model=model_name,
        tokenizer=tokenizer,
        trust_remote_code=True,
        max_model_len=5120,
        tensor_parallel_size=num_gpus,
    )
    
    if "tmp_model" in model_name:
        # remove the tmp_model dir
        shutil.rmtree(os.path.join(model_dir, "tmp_model"))

    if is_gsm_8k:
        dataset = load_dataset(dataset_name, "main", trust_remote_code=True)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported")
    if is_eval:
        dataset = dataset["test"]
    else:
        dataset = dataset["train"]
    
    if num_samples is not None:
        logger.info(f"Sampling {num_samples} samples from the dataset")
        dataset = dataset.select(range(num_samples))
        save_file = f"distillation_data-{num_samples}.jsonl" if not is_eval else f"eval-{num_samples}.jsonl"
    elif num_splits is not None:
        save_file = f"distillation_data_split-{split_id}-of-{num_splits}.jsonl"
    else:
        save_file = "distillation_data.jsonl" if not is_eval else "eval.jsonl"

    if is_gsm_8k:
        preprocess_fn = partial(batch_preprocess_fn, task="chat-gen-gsm8k")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported")
    dataset = dataset.map(preprocess_fn, batched=True, num_proc=num_workers, remove_columns=dataset.column_names)
    
    if num_splits is not None:
        num_samples_per_split = len(dataset) // num_splits
        logger.info(f"Selecting {num_samples_per_split} samples for split {split_id} of {num_splits}")
        dataset = dataset.select(range(split_id * num_samples_per_split, (split_id + 1) * num_samples_per_split))
        

    with open(os.path.join(save_dir, save_file), 'a') as file:
        for j, messages in enumerate(tqdm(dataset["content"], desc="Generating distillation data via API")):
            outputs = llm.chat(messages, sampling_params=sampling_params)
            
            response = outputs[0].outputs[0].text
            prompt = messages[-1]["content"]
            result_content = json.dumps({
                "response": response,
                "prompt": prompt,
            }, ensure_ascii=False) + "\n"
            file.write(result_content)


if __name__ == "__main__":
    Fire(api_generate_distillation_data_eager)
