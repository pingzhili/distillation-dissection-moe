export PYTHONPATH=$PYTHONPATH:src
export MODEL_PATH="outputs/qwen3-8b-antidistill-coef0.00003-temp2-head_proj0-epoch1-lr5e-5-2xproxy/checkpoint-60/lm_head.pt"


CUDA_VISIBLE_DEVICES=0 python scripts/generate-eval.py \
    --model_path="$MODEL_PATH" --task_name="arcc" &> logs/qwen3-8b-antidistill-mixed-2xproxy-arcc.log &

sleep 120

CUDA_VISIBLE_DEVICES=4 python scripts/generate-eval.py \
    --model_path="$MODEL_PATH" --task_name="csqa" &> logs/qwen3-8b-antidistill-mixed-2xproxy-csqa.log &

sleep 120

CUDA_VISIBLE_DEVICES=7 python scripts/generate-eval.py \
    --model_path="$MODEL_PATH" --task_name="gsm8k" &> logs/qwen3-8b-antidistill-mixed-2xproxy-gsm8k.log &

sleep 120