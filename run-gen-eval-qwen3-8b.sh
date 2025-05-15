export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=4


MODEL_PATH="outputs/qwen3-8b-antidistill-coef0.00003-temp2-head_proj0-epoch1-lr5e-5/checkpoint-120/lm_head.pt"
python scripts/generate-eval.py \
    --model_path="$MODEL_PATH" --task_name="arcc"