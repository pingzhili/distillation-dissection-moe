import json
from functools import partial
from loguru import logger
from tqdm import tqdm
import torch
from fire import Fire

from ddmoe.evaluation import evaluate_predictions
from ddmoe.data import batch_preprocess_fn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
def main(
    model_path: str,
    task_name: str="gsm8k",
    num_workers: int=4,
):
    if task_name == "gsm8k":
        dataset_name = "openai/gsm8k"
    else:
        raise ValueError(f"Task name {task_name} not supported")
    
    if "llama-3.2" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        preprocess_fn = partial(batch_preprocess_fn, task="reasoning-llama-3.2-eval", tokenizer=tokenizer)
    else:
        raise ValueError(f"Model {model_path} not supported")
    
    datasets = load_dataset(dataset_name, "main", trust_remote_code=True)["test"]
    datasets = datasets.map(preprocess_fn, num_proc=num_workers)
    
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cuda:0")
    model.eval()
    
    predictions = []
    ground_truths = []
    
    for i in tqdm(range(len(datasets))):
        input_ids = datasets[i]["input_ids"]
        answer = datasets[i]["answer"]
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=8192)
            predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
            ground_truths.append(answer)

    