export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1

SOURCE_MODEL="qwen-antidistill-coef0.00001-temp2-epoch2-lr5e-5-checkpoint-60"
accelerate launch --config_file configs/zero3-4gpu-ga8.yaml \
    --num_processes=4 \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_port=23333 \
    --mixed_precision=fp16 \
    scripts/finetune-sft.py \
    --base_model_name="meta-llama/Llama-3.2-1B" \
    --output_dir="outputs/llama-3.2-1b-distill--$SOURCE_MODEL" \
    --dataset_name="data/antidistill-exps/gsm8k/$SOURCE_MODEL.jsonl" \
    --num_train_epochs=3 \
    --batch_size_per_device=4 \
    --gradient_accumulation_steps=8 \
    --debugging=True