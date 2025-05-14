export PYTHONPATH=$PYTHONPATH:src
export NCCL_P2P_DISABLE=1
# export CUDA_VISIBLE_DEVICES=7

# MODEL_PATH="outputs/qwen7b-antidistill-coef0.00003-temp2-epoch2-lr5e-5/checkpoint-60"
# python scripts/generate-distillation-data-offline.py \
#     --model_name="$MODEL_PATH" \
#     --dataset_name="openai/gsm8k" \
#     --save_dir="$MODEL_PATH/gen-gsm8k"


# MODEL_PATH="outputs/llama-3.2-1b-distill--qwen-antidistill-coef0.00003-temp2-epoch2-lr5e-5-checkpoint-60/checkpoint-177"
# python scripts/generate-distillation-data-offline.py \
#     --model_name="$MODEL_PATH" \
#     --dataset_name="openai/gsm8k" \
#     --save_dir="$MODEL_PATH/gen-gsm8k" \
#     --is_eval


# # testing
# for dir in outputs/qwen7b-antidistill-coef0.0001-temp2-epoch2-lr5e-5/*; do
#     MODEL_PATH="$dir"
#     python scripts/generate-distillation-data-offline.py \
#         --model_name="$MODEL_PATH" \
#         --dataset_name="openai/gsm8k" \
#         --save_dir="$MODEL_PATH/gen-gsm8k" \
#         --num_samples=10
# done

# for dir in outputs/qwen7b-antidistill-coef0.0003-temp2-epoch2-lr5e-5/*; do
#     MODEL_PATH="$dir"
#     python scripts/generate-distillation-data-offline.py \
#         --model_name="$MODEL_PATH" \
#         --dataset_name="openai/gsm8k" \
#         --save_dir="$MODEL_PATH/gen-gsm8k" \
#         --num_samples=10
# done

# for dir in outputs/qwen7b-antidistill-coef0.00001-temp2-epoch2-lr5e-5/*; do
#     MODEL_PATH="$dir"
#     python scripts/generate-distillation-data-offline.py \
#         --model_name="$MODEL_PATH" \
#         --dataset_name="openai/gsm8k" \
#         --save_dir="$MODEL_PATH/gen-gsm8k" \
#         --num_samples=10
# done

# for dir in outputs/qwen7b-antidistill-coef0.00003-temp2-epoch2-lr5e-5/*; do
#     MODEL_PATH="$dir"
#     python scripts/generate-distillation-data-offline.py \
#         --model_name="$MODEL_PATH" \
#         --dataset_name="openai/gsm8k" \
#         --save_dir="$MODEL_PATH/gen-gsm8k" \
#         --num_samples=10
# done

# CHECKPOINT=120
# CUDA_VISIBLE_DEVICES=5 python scripts/generate-distillation-data-offline.py \
#     --model_name="outputs/qwen3-8b-antidistill-coef0.00001-temp2-head_proj0-epoch1-lr5e-5/checkpoint-$CHECKPOINT/lm_head.pt" \
#     --dataset_name="openai/gsm8k" \
#     --save_dir="outputs/qwen3-8b-antidistill-coef0.00001-temp2-head_proj0-epoch1-lr5e-5/checkpoint-$CHECKPOINT/gen-gsm8k" 

CUDA_VISIBLE_DEVICES=6,7 python scripts/generate-distillation-data-offline.py \
    --model_name="outputs/qwen3-8b-antidistill-coef0.00001-temp2-head_proj0-epoch1-lr5e-5/checkpoint-$CHECKPOINT/lm_head.pt" \
    --dataset_name="openai/gsm8k" \
    --save_dir="outputs/qwen3-8b-antidistill-coef0.00001-temp2-head_proj0-epoch1-lr5e-5/checkpoint-$CHECKPOINT/gen-gsm8k" \
    --num_gpus=2 --num_splits=8 --split_id=0