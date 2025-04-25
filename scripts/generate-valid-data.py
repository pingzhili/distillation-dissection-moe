from datasets import load_dataset
from vllm import LLM
from vllm.sampling_params import SamplingParams
from fire import Fire
import pandas as pd
from tqdm import tqdm
from loguru import logger


def dump_valid_data(
    model_name: str, 
    save_dir: str
):
    tokenizer = "allenai/OLMoE-1B-7B-0125-Instruct" if "olmoe" in model_name else "moonshotai/Moonlight-16B-A3B-Instruct"
    sampling_params = SamplingParams(
        max_tokens=1024
    )
    llm = LLM(
        model=model_name,
        tokenizer=tokenizer,
        max_model_len=4096,
    )
    
    dataset = load_dataset("Phando/sft-dataset-valid", split="train", trust_remote_code=True)
    question_list = []
    response_list = []
    source_list = []
    
    for example in tqdm(dataset, desc="Generating responses..."):
        question = example["question"]
        source = example["source"]
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]
        
        outputs = llm.chat(messages, sampling_params=sampling_params)
        
        response = outputs[0].outputs[0].text
        logger.info(f"Question: {question}")
        logger.info(f"Response: {response}")
        logger.info(f"Source: {source}")

        question_list.append(question)
        response_list.append(response)
        source_list.append(source)
        
    
    df = pd.DataFrame({
        "question": question_list,
        "response": response_list,
        "source": source_list,
    })
    df.to_csv(save_dir, index=False)
    logger.info(f"Saved to {save_dir}")
    
    
if __name__ == "__main__":
    Fire(dump_valid_data)
    