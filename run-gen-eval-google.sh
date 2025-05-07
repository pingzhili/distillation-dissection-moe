export PYTHONPATH=$PYTHONPATH:src
export MODEL_PATH="../ddmoe/outputs/llama-3.2-1b-distill--qwen-antidistill-coef0.00001-temp2-epoch2-lr5e-5-checkpoint-60/checkpoint-177"

CUDA_VISIBLE_DEVICES=0,1 python scripts/generate-eval.py \
    --model_path="$MODEL_PATH" --task_name="tabmwp" --num_gpus=2 &

CUDA_VISIBLE_DEVICES=2,3 python scripts/generate-eval.py \
    --model_path="$MODEL_PATH" --task_name="math" --num_gpus=2 &

# CUDA_VISIBLE_DEVICES=4,5 python scripts/generate-eval.py \
#     --model_path="$MODEL_PATH" --task_name="arcc" --num_gpus=2 &

# CUDA_VISIBLE_DEVICES=6,7 python scripts/generate-eval.py \
#     --model_path="$MODEL_PATH" --task_name="csqa" --num_gpus=2 &
