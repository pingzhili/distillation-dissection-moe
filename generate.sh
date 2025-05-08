export PYTHONPATH=$PYTHONPATH:src

#python scripts/generate-distillation-data.py
# python scripts/generate-distillation-data.py \
#   --dataset_name="openai/gsm8k" --save_dir="data/r1-qwen-7b-gsm8k/" \
#   --model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

python scripts/generate-distillation-data.py \
  --dataset_name="openai/gsm8k" --save_dir="data/qwen3-8b-gsm8k-v2/" \
  --model_name="Qwen/Qwen3-8B" --base_url="http://localhost:2333/v1"


# python scripts/generate-distillation-data.py \
#   --dataset_name="ServiceNow-AI/R1-Distill-SFT" --save_dir="data/r1-qwen-7b-mixed/" \
#   --model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --max_tokens=8192