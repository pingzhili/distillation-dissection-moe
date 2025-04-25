export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1

KD_COEF=${1:-0.001}
KD_TEMP=2
EPOCH=2
LR=5e-5
OUTPUT_DIR="outputs/qwen7b-antidistill-coef$KD_COEF-temp$KD_TEMP-epoch$EPOCH-lr$LR"

accelerate launch --config_file configs/zero3-4gpu-ga32.yaml --main_process_port=23333 \
    scripts/finetune-antidistill.py \
    --anti_kd_coef=$KD_COEF \
    --kd_temperature=$KD_TEMP \
    --output_dir=$OUTPUT_DIR \
    --num_train_epochs=$EPOCH \
    --batch_size_per_device=1 \
    --gradient_accumulation_steps=32 \
    --checkpointing_steps=10 \
    --learning_rate=$LR
#    --debugging=True