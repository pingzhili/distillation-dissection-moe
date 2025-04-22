export PYTHONPATH=$PYTHONPATH:src

#python scripts/generate-distillation-data.py
python scripts/generate-distillation-data.py \
  --dataset_name="openai/gsm8k" --save_dir="data/r1-qwen-7b-gsm8k/" \
  --model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"