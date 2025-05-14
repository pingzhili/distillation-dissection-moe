export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=1,2,4,5,6,7
export NCCL_P2P_DISABLE=1

KD_COEF=${1:-0.00003}
HEAD_PROJ_DIM=${2:-0}
KD_TEMP=2
EPOCH=1
LR=5e-5
# OUTPUT_DIR="outputs/qwen7b-antidistill-coef$KD_COEF-temp$KD_TEMP-head_proj$HEAD_PROJ_DIM-epoch$EPOCH-lr$LR"
OUTPUT_DIR="outputs/qwen3-8b-antidistill-coef$KD_COEF-temp$KD_TEMP-head_proj$HEAD_PROJ_DIM-epoch$EPOCH-lr$LR-2xproxy"

accelerate launch --config_file configs/zero3-6gpu-ga21.yaml --main_process_port=23333 \
    scripts/finetune-antidistill.py \
    --teacher_model_name="Qwen/Qwen3-8B" \
    --proxy_model_name="Qwen/Qwen3-4B,Qwen/Qwen3-1.7B" \
    --dataset_path="pingzhili/qwen3-8b-mixed" \
    --anti_kd_coef=$KD_COEF \
    --kd_temperature=$KD_TEMP \
    --output_dir=$OUTPUT_DIR \
    --num_train_epochs=$EPOCH \
    --batch_size_per_device=1 \
    --gradient_accumulation_steps=21 \
    --checkpointing_steps=60 \
    --learning_rate=$LR \
    --lm_head_projector_dim=$HEAD_PROJ_DIM \
    --max_length=4096
#    --debugging=True