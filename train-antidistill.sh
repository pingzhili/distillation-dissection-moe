export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1

KD_COEF=0.1
KD_TEMP=1.0
EPOCH=3
OUTPUT_DIR="outputs/olmoe-antidistill-coef$KD_COEF-temp$KD_TEMP-epoch$EPOCH"

accelerate launch --config_file configs/zero3-4gpu-ga2.yaml \
    --num_processes=4 \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_port=23333 \
    --mixed_precision=fp16 \
    scripts/finetune-antidistill.py \
    --output_dir=$OUTPUT_DIR \
    --num_train_epochs=$EPOCH \
    --batch_size_per_device=16 \
    --gradient_accumulation_steps=2 \
    --debugging=True