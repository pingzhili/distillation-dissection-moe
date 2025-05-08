export PYTHONPATH=$PYTHONPATH:src
# export MODEL_PATH="../ddmoe/outputs/llama-3.2-1b-distill--qwen-antidistill-coef0.00001-temp2-epoch2-lr5e-5-checkpoint-60/checkpoint-177"

# CUDA_VISIBLE_DEVICES=0,1 python scripts/generate-eval.py \
#     --model_path="$MODEL_PATH" --task_name="tabmwp" --num_gpus=2 &

# CUDA_VISIBLE_DEVICES=2,3 python scripts/generate-eval.py \
#     --model_path="$MODEL_PATH" --task_name="math" --num_gpus=2 &

# CUDA_VISIBLE_DEVICES=4,5 python scripts/generate-eval.py \
#     --model_path="$MODEL_PATH" --task_name="arcc" --num_gpus=2 &

# CUDA_VISIBLE_DEVICES=6,7 python scripts/generate-eval.py \
#     --model_path="$MODEL_PATH" --task_name="csqa" --num_gpus=2 &

# CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-8B \
#     --port 23333 --tensor-parallel-size 2 --enable-reasoning --reasoning-parser deepseek_r1 &

# CUDA_VISIBLE_DEVICES=2,3 vllm serve Qwen/Qwen3-8B \
#     --port 23334 --tensor-parallel-size 2 --enable-reasoning --reasoning-parser deepseek_r1 &

# CUDA_VISIBLE_DEVICES=4,5 vllm serve Qwen/Qwen3-8B \
#     --port 23335 --tensor-parallel-size 2 --enable-reasoning --reasoning-parser deepseek_r1 &

# CUDA_VISIBLE_DEVICES=6,7 vllm serve Qwen/Qwen3-8B \
#     --port 23336 --tensor-parallel-size 2 --enable-reasoning --reasoning-parser deepseek_r1 &

# CUDA_VISIBLE_DEVICES=8,9 vllm serve Qwen/Qwen3-8B \
#     --port 23337 --tensor-parallel-size 2 --enable-reasoning --reasoning-parser deepseek_r1 &

# CUDA_VISIBLE_DEVICES=10,11 vllm serve Qwen/Qwen3-8B \
#     --port 23338 --tensor-parallel-size 2 --enable-reasoning --reasoning-parser deepseek_r1 &

# CUDA_VISIBLE_DEVICES=12,13 vllm serve Qwen/Qwen3-8B \
#     --port 23339 --tensor-parallel-size 2 --enable-reasoning --reasoning-parser deepseek_r1 &

# CUDA_VISIBLE_DEVICES=14,15 vllm serve Qwen/Qwen3-8B \
#     --port 23340 --tensor-parallel-size 2 --enable-reasoning --reasoning-parser deepseek_r1 &

# # wait for 10 mins until all the servers are started
# sleep 600

# distributed generate by 8 splits
for i in {0..7}; do
    PORT=$((23333 + i))
    python scripts/generate-distillation-data.py \
        --dataset_name="ServiceNow-AI/R1-Distill-SFT" --save_dir="data/r1-qwen-7b-mixed/" \
        --model_name="Qwen/Qwen3-8B" --max_tokens=8192 \
        --shuffle=False --split_id=${i} --num_splits=8 --base_url="http://localhost:${PORT}/v1" &
done


