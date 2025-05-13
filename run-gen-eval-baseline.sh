export PYTHONPATH=$PYTHONPATH:src

CUDA_VISIBLE_DEVICES=0 python scripts/generate-eval.py \
    --model_path="Qwen/Qwen3-8B" --task_name="csqa" &> qwen3-8b-csqa.log &

CUDA_VISIBLE_DEVICES=1 python scripts/generate-eval.py \
    --model_path="Qwen/Qwen3-8B" --task_name="math" &> qwen3-8b-math.log &

CUDA_VISIBLE_DEVICES=2 python scripts/generate-eval.py \
    --model_path="Qwen/Qwen3-8B" --task_name="gsm8k" &> qwen3-8b-gsm8k.log &

CUDA_VISIBLE_DEVICES=3 python scripts/generate-eval.py \
    --model_path="Qwen/Qwen3-8B" --task_name="arcc" &> qwen3-8b-arcc.log &

CUDA_VISIBLE_DEVICES=4 python scripts/generate-eval.py \
    --model_path="meta-llama/Llama-3.2-1B-Instruct" --task_name="csqa" &> llama-3.2-1b-csqa.log &

CUDA_VISIBLE_DEVICES=5 python scripts/generate-eval.py \
    --model_path="google/gemma-3-1b-it" --task_name="csqa" &> gemma-3-1b-csqa.log &

CUDA_VISIBLE_DEVICES=6 python scripts/generate-eval.py \
    --model_path="google/gemma-3-1b-it" --task_name="math" &> gemma-3-1b-math.log &

CUDA_VISIBLE_DEVICES=7 python scripts/generate-eval.py \
    --model_path="google/gemma-3-1b-it" --task_name="gsm8k" &> gemma-3-1b-gsm8k.log &

