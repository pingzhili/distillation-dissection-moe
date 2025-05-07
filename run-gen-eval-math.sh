export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=5
export CUDA_LAUNCH_BLOCKING=1

# MODEL_PATH="outputs/qwen7b-antidistill-coef0.00003-temp2-epoch2-lr5e-5/checkpoint-60"
# python scripts/generate-distillation-data-offline.py \
#     --model_name="$MODEL_PATH" \
#     --dataset_name="openai/gsm8k" \
#     --save_dir="$MODEL_PATH/gen-gsm8k"


MODEL_PATH="outputs/llama-3.2-1b-distill--qwen-antidistill-coef0.00003-temp2-epoch2-lr5e-5-checkpoint-60/checkpoint-177"
python scripts/generate-eval.py \
    --model_path="$MODEL_PATH" --task_name="math"

MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
python scripts/generate-eval.py \
    --model_path="$MODEL_PATH" --task_name="math"


MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
python scripts/generate-eval.py \
    --model_path="$MODEL_PATH" --task_name="math"


MODEL_PATH="outputs/qwen7b-antidistill-coef0.00003-temp2-epoch2-lr5e-5/checkpoint-60"
python scripts/generate-eval.py \
    --model_path="$MODEL_PATH" --task_name="math"




# MODEL_PATH="../ddmoe/outputs/llama-3.2-1b-distill--qwen-antidistill-coef0.00001-temp2-epoch2-lr5e-5-checkpoint-60/checkpoint-177"
# python scripts/generate-eval.py \
#     --model_path="$MODEL_PATH" --num_gpus=2

# MODEL_PATH="outputs/qwen7b-antidistill-coef0.00001-temp2-epoch2-lr5e-5/checkpoint-60"
# python scripts/generate-eval.py \
#     --model_path="$MODEL_PATH" 

