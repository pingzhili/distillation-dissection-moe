export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_LAUNCH_BLOCKING=1
#export TORCH_USE_CUDA_DSA=1
# python debug.py
#python scripts/generate-distillation-data.py
#accelerate launch --num_processes=8 scripts/generate-distillation-data.py
#sudo docker run -e NCCL_P2P_DISABLE=1 --gpus all --shm-size 32g -p 0.0.0.0:23333:23333 -v ~/.cache/huggingface:/root/.cache/huggingface -v /home/pingzhi/model-checkpoints:/model-checkpoints --ipc=host --network=host --name moonlight --privileged lmsysorg/sglang:v0.4.3.post2-cu125 bash -c "pip install blobfile tiktoken transformers==4.48.2 && python3 -m sglang.launch_server --model-path moonshotai/Moonlight-16B-A3B-Instruct --trust-remote-code --disable-radix-cache --tp 2 --dp 4 2>&1" | sudo tee output.log

accelerate launch --config_file configs/zero-3-offload.yaml scripts/finetune.py \
  --base_model_name="allenai/OLMoE-1B-7B-0125" --dataset_name="Phando/sft-dataset-from-moonlight-filtered" \
  --output_dir="outputs/olmoe-1b-7b-0125-sft"
#accelerate launch --config_file configs/zero-3.yaml scripts/finetune.py \
#  --base_model_name="deepseek-ai/DeepSeek-V2-Lite" --dataset_name="Phando/sft-dataset-from-moonlight-filtered" \
#  --output_dir="outputs/deepseek-v2-lite-sft" --batch_size_per_device=1 --gradient_accumulation_steps=16 \
#  --checkpointing_steps=200 --enable_lora=True