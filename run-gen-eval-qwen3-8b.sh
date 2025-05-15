export PYTHONPATH=$PYTHONPATH:src
export MODEL_PATH="outputs/qwen3-8b-antidistill-coef0.00003-temp2-head_proj0-epoch1-lr5e-5/checkpoint-60/lm_head.pt"


CUDA_VISIBLE_DEVICES=3 python scripts/generate-eval.py \
    --model_path="$MODEL_PATH" --task_name="arcc" &> logs/qwen3-8b-antidistill-mixed-arcc.log &

sleep 300

CUDA_VISIBLE_DEVICES=5 python scripts/generate-eval.py \
    --model_path="$MODEL_PATH" --task_name="csqa" &> logs/qwen3-8b-antidistill-mixed-csqa.log &

# sleep 300

# CUDA_VISIBLE_DEVICES=7 python scripts/generate-eval.py \
#     --model_path="$MODEL_PATH" --task_name="gsm8k" &> logs/qwen3-8b-antidistill-mixed-gsm8k.log &

# sleep 300