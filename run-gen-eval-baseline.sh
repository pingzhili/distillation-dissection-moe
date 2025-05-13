export PYTHONPATH=$PYTHONPATH:src

CUDA_VISIBLE_DEVICES=0 python scripts/generate-eval.py \
    --model_path="Qwen/Qwen3-8B" --task_name="csqa" &> csqa.log &

CUDA_VISIBLE_DEVICES=2 python scripts/generate-eval.py \
    --model_path="Qwen/Qwen3-8B" --task_name="math" &> math.log &

CUDA_VISIBLE_DEVICES=3 python scripts/generate-eval.py \
    --model_path="Qwen/Qwen3-8B" --task_name="gsm8k" &> gsm8k.log &

CUDA_VISIBLE_DEVICES=4 python scripts/generate-eval.py \
    --model_path="Qwen/Qwen3-8B" --task_name="arcc" &> arcc.log &