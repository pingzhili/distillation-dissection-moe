export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=4


MODEL_PATH="outputs/llama-3.2-1b-distill--qwen-antidistill-coef0.00003-temp2-epoch2-lr5e-5-checkpoint-60/checkpoint-177"
python scripts/generate-eval.py \
    --model_path="$MODEL_PATH" --task_name="tabmwp"