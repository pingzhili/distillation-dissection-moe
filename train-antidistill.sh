export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1

KD_COEF=0.1
KD_TEMP=1.0
EPOCH=3
OUTPUT_DIR="outputs/qwen7b-antidistill-coef$KD_COEF-temp$KD_TEMP-epoch$EPOCH"

accelerate launch --config_file configs/zero3-2gpu-ga16.yaml \
    --num_processes=2 \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_port=23333 \
    --mixed_precision=fp16 \
    scripts/finetune-antidistill.py \
    --anti_kd_coef=$KD_COEF \
    --kd_temperature=$KD_TEMP \
    --output_dir=$OUTPUT_DIR \
    --num_train_epochs=$EPOCH \
    --batch_size_per_device=4 \
    --gradient_accumulation_steps=16 \
    --debugging=True