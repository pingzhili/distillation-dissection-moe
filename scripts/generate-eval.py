import json
from functools import partial
from loguru import logger
from tqdm import tqdm
from fire import Fire
import os

from vllm import LLM
from vllm.sampling_params import SamplingParams
from ddmoe.evaluation import evaluate_predictions
from ddmoe.data import batch_preprocess_fn
from transformers import AutoTokenizer
from datasets import load_dataset


def main(
    model_path: str,
    task_name: str="gsm8k",
    batch_size: int=4,
    num_workers: int=4,
):
    if task_name == "gsm8k":
        dataset_name = "openai/gsm8k"
    else:
        raise ValueError(f"Task name {task_name} not supported")
    
    if "llama-3.2" in model_path.lower():
        tokenizer_name = "meta-llama/Llama-3.2-1B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        preprocess_fn = partial(batch_preprocess_fn, task="reasoning-llama-3.2-eval", tokenizer=tokenizer)
    elif "r1-distill-qwen-7b" in model_path.lower() or "qwen7b-antidistill" in model_path.lower():
        tokenizer_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        preprocess_fn = partial(batch_preprocess_fn, task="reasoning-llama-3.2-eval", tokenizer=tokenizer) # this is not typo
    else:
        raise ValueError(f"Model {model_path} not supported")
    
    # no cache
    datasets = load_dataset(dataset_name, "main", trust_remote_code=True)["test"]
    datasets = datasets.map(
        preprocess_fn, 
        num_proc=num_workers, 
        batched=True
    )
    # debugging
    # datasets = datasets.select(range(4))
    
    sampling_params = SamplingParams(max_tokens=8192, stop=[tokenizer.pad_token])
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_name,
        trust_remote_code=True,
        max_model_len=8192,
    )
    # model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cuda:0")
    # model.eval()
    
    predictions = []
    ground_truths = []
    prompt_list = []
    
    logger.info(f"Generating predictions for {len(datasets)} examples with batch size {batch_size}!")
    for batch_id in tqdm(range(len(datasets) // batch_size)):
        batch_datasets = datasets[batch_id * batch_size:(batch_id + 1) * batch_size]
        batch_prompt = [prompt.split(tokenizer.bos_token)[-1] for prompt in batch_datasets['prompt']]
        batch_answer = batch_datasets['response']
        
        outputs = llm.generate(batch_prompt, sampling_params=sampling_params)
        batch_prediction = [output.outputs[0].text for output in outputs]
        
        predictions.extend(batch_prediction)
        ground_truths.extend(batch_answer)
        prompt_list.extend(batch_prompt)
        
        for pred, gt in zip(batch_prediction, batch_answer):
            logger.debug(f"Input: {batch_prompt}")
            logger.debug(f"Prediction: {pred}")
            logger.debug(f"Ground truth: {gt}")
            logger.debug("-"*100)
    
    # for i in tqdm(range(len(datasets))):
    #     prompt = datasets[i]["prompt"]
    #     # remove bos token from prompt
    #     prompt = prompt.split(tokenizer.bos_token)[-1]
    #     answer = datasets[i]["response"]
        
    #     outputs = llm.generate(prompt, sampling_params=sampling_params)
    #     prediction = outputs[0].outputs[0].text
        
    #     predictions.append(prediction)
    #     ground_truths.append(answer)
    #     logger.debug(f"Input: {prompt}")
    #     logger.debug(f"Prediction: {prediction}")
    
    results = evaluate_predictions(predictions, ground_truths)
    logger.info(f"Results: {results}")
    
    results["model_name"] = model_path
    results["task_name"] = task_name
    results["content"] = []
    for sample_id in range(len(predictions)):
        results["content"].append({
            "id": sample_id,
            "prompt": prompt_list[sample_id],
            "prediction": predictions[sample_id],
            "ground_truth": ground_truths[sample_id]
        })
    
    save_dir = model_path if "outputs/" in model_path else os.path.join("outputs", model_path)
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    with open(os.path.join(save_dir, f"{task_name}-results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    Fire(main)
