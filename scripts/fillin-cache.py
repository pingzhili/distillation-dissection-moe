from datasets import load_dataset
from transformers import AutoModelForCausalLM

load_dataset("ServiceNow-AI/R1-Distill-SFT", "v1", trust_remote_code=True)
load_dataset("allenai/tulu-3-hard-coded-10x", split="train")

AutoModelForCausalLM.from_pretrained("moonshotai/Moonlight-16B-A3B-Instruct", trust_remote_code=True)
AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0125", trust_remote_code=True)
AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0125-Instruct", trust_remote_code=True)
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-32B-Instruct", trust_remote_code=True)
AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
