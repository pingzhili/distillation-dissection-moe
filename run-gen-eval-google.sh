export PYTHONPATH=$PYTHONPATH:src
export NCCL_P2P_DISABLE=1
# export MODEL_PATH="../ddmoe/outputs/llama-3.2-1b-distill--qwen-antidistill-coef0.00001-temp2-epoch2-lr5e-5-checkpoint-60/checkpoint-177"

# CUDA_VISIBLE_DEVICES=0,1 python scripts/generate-eval.py \
#     --model_path="$MODEL_PATH" --task_name="tabmwp" --num_gpus=2 &

# CUDA_VISIBLE_DEVICES=2,3 python scripts/generate-eval.py \
#     --model_path="$MODEL_PATH" --task_name="math" --num_gpus=2 &

# CUDA_VISIBLE_DEVICES=4,5 python scripts/generate-eval.py \
#     --model_path="$MODEL_PATH" --task_name="arcc" --num_gpus=2 &

# CUDA_VISIBLE_DEVICES=6,7 python scripts/generate-eval.py \
#     --model_path="$MODEL_PATH" --task_name="csqa" --num_gpus=2 &

#########################

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

# # distributed generate by 8 splits
# for i in {0..7}; do
#     PORT=$((23333 + i))
#     python scripts/generate-distillation-data.py \
#         --dataset_name="ServiceNow-AI/R1-Distill-SFT" --save_dir="data/r1-qwen-7b-mixed/" \
#         --model_name="Qwen/Qwen3-8B" --max_tokens=8192 \
#         --shuffle=False --split_id=${i} --num_splits=8 --base_url="http://localhost:${PORT}/v1" &
# done


#########################


# for KDCOEF in 0.00003 0.00001;do
#   bash train-antidistill.sh $KDCOEF
# done

# for i in {0..7}; do
#     # devices are i*2, i*2+1
#     CUDA_VISIBLE_DEVICES=$((i*2)),$((i*2+1)) python scripts/generate-distillation-data-offline.py \
#       --model_name="outputs/qwen3-8b-antidistill-coef0.00003-temp2-head_proj0-epoch1-lr5e-5/checkpoint-120/lm_head.pt" \
#       --dataset_name="openai/gsm8k" \
#       --save_dir="outputs/qwen3-8b-antidistill-coef0.00003-temp2-head_proj0-epoch1-lr5e-5/checkpoint-120/gen-gsm8k" \
#       --num_gpus=2 --num_splits=8 --split_id=${i} &> "logs/qwen3-8b-antidistill-coef0.00003-temp2-head_proj0-epoch1-lr5e-5--checkpoint-120-gen-gsm8k-${i}.log" &
#     sleep 600
# done

# CUDA_VISIBLE_DEVICES=$((i*2)),$((i*2+1)) python scripts/generate-eval.py \
#       --model_path="outputs/qwen3-8b-antidistill-coef0.00003-temp2-head_proj0-epoch1-lr5e-5/checkpoint-120/lm_head.pt" \
#       --task_name="gsm8k"&> "logs/qwen3-8b-antidistill-coef0.00003-temp2-head_proj0-epoch1-lr5e-5--checkpoint-120-gen-gsm8k-${i}.log" &
# sleep 600

SOURCE_MODEL="qwen3-8b-antidistill-coef0.00003-temp2-head_proj0-epoch1-lr5e-5-checkpoint-120"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file configs/zero3-8gpu-ga16.yaml \
    --num_processes=8 \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_port=23333 \
    --mixed_precision=bf16 \
    scripts/finetune-sft.py \
    --base_model_name="meta-llama/Llama-3.2-1B" \
    --output_dir="outputs/llama-3.2-1b-distill--$SOURCE_MODEL" \
    --dataset_name="data/antidistill-exps/gsm8k/$SOURCE_MODEL.jsonl" \
    --num_train_epochs=3 \
    --batch_size_per_device=1 \
    --gradient_accumulation_steps=16 &
  
SOURCE_MODEL="qwen3-8b-antidistill-coef0.00003-temp2-head_proj0-epoch1-lr5e-5-checkpoint-120"
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 accelerate launch --config_file configs/zero3-8gpu-ga16.yaml \
    --num_processes=8 \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_port=23333 \
    --mixed_precision=bf16 \
    scripts/finetune-sft.py \
    --base_model_name="google/gemma-3-1b-it" \
    --output_dir="outputs/gemma-3-1b-it-distill--$SOURCE_MODEL" \
    --dataset_name="data/antidistill-exps/gsm8k/$SOURCE_MODEL.jsonl" \
    --num_train_epochs=3 \
    --batch_size_per_device=1 \
    --gradient_accumulation_steps=16 &