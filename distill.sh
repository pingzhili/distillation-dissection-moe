export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1
#export TORCH_DISTRIBUTED_TIMEOUT=7200000
#export NCCL_BLOCKING_WAIT=0
#export CUDA_LAUNCH_BLOCKING=1
#export TORCH_USE_CUDA_DSA=1
# python debug.py
#python scripts/generate-distillation-data.py
#accelerate launch --num_processes=8 scripts/generate-distillation-data.py
#sudo docker run -e NCCL_P2P_DISABLE=1 --gpus all --shm-size 32g -p 0.0.0.0:23333:23333 -v ~/.cache/huggingface:/root/.cache/huggingface -v /home/pingzhi/model-checkpoints:/model-checkpoints --ipc=host --network=host --name moonlight --privileged lmsysorg/sglang:v0.4.3.post2-cu125 bash -c "pip install blobfile tiktoken transformers==4.48.2 && python3 -m sglang.launch_server --model-path moonshotai/Moonlight-16B-A3B-Instruct --trust-remote-code --disable-radix-cache --tp 2 --dp 4 2>&1" | sudo tee output.log

#accelerate launch --config_file configs/zero-3-offload.yaml scripts/finetune-sft.py \
#  --base_model_name="allenai/OLMoE-1B-7B-0125" --dataset_name="Phando/sft-dataset-from-moonlight-filtered" \
#  --output_dir="outputs/olmoe-1b-7b-0125-sft-filtered"

#accelerate launch --config_file configs/zero-3-offload.yaml scripts/finetune-sft.py \
#   --base_model_name="allenai/OLMoE-1B-7B-0125" --dataset_name="Phando/sft-dataset-original-filtered" \
#   --output_dir="outputs/olmoe-1b-7b-0125-sft-original-filtered-debug"

#accelerate launch --config_file configs/zero-3-offload.yaml scripts/finetune-sft.py \
#  --base_model_name="moonshotai/Moonlight-16B-A3B-Instruct" --dataset_name="Phando/sft-r1-distill" \
#  --output_dir="outputs/moonlight-instruct-sft-r1-distill"

#  open-thought source: {'riddle_sense', 'camelai_biology', 'camelai_chemistry', 'taco', 'code_contests', 'codeforces', 'camelai_physics', 'apps', 'numina_math'}
for SUB_TASK in "riddle_sense" "camelai_biology" "camelai_chemistry" "camelai_physics"; do
  accelerate launch --config_file configs/zero3-4gpu-ga4.yaml \
    --num_processes=4 \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_port=23333 \
    --mixed_precision=fp16 \
    scripts/finetune-sft.py \
    --output_dir=$OUTPUT_DIR \
    --base_model_name="allenai/OLMoE-1B-7B-0125" \
    --output_dir="outputs/olmoe-sft-$SUB_TASK" \
    --dataset_name="Phando/OpenThoughts-114k-R1-Distill" \
    --dataset_filter_condition="example['source'] == '$SUB_TASK'" \
    --num_train_epochs=3 \
    --batch_size_per_device=8 \
    --gradient_accumulation_steps=4
done

for SUB_TASK in "taco" "code_contests" "codeforces" "apps" "numina_math"; do
  accelerate launch --config_file configs/zero3-4gpu-ga8.yaml \
    --num_processes=4 \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_port=23333 \
    --mixed_precision=fp16 \
    scripts/finetune-sft.py \
    --output_dir=$OUTPUT_DIR \
    --base_model_name="allenai/OLMoE-1B-7B-0125" \
    --output_dir="outputs/olmoe-sft-$SUB_TASK" \
    --dataset_name="Phando/OpenThoughts-114k-R1-Distill" \
    --dataset_filter_condition="example['source'] == '$SUB_TASK'" \
    --num_train_epochs=3 \
    --batch_size_per_device=4 \
    --gradient_accumulation_steps=8
done