export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=7

MODEL_PATH="outputs/qwen7b-antidistill-coef0.00003-temp2-epoch2-lr5e-5/checkpoint-60"
python scripts/generate-distillation-data-offline.py \
    --model_name="$MODEL_PATH" \
    --dataset_name="openai/gsm8k" \
    --save_dir="$MODEL_PATH/gen-gsm8k"


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


