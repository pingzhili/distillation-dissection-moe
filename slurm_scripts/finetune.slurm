#!/bin/bash
#SBATCH -J Distill-R1          # Job name
#SBATCH -o Distill-R1%j.out      # Output file name
#SBATCH -e Distill-R1%j.err      # Error file name
#SBATCH -p gh                   # Queue (partition) name
#SBATCH -N 8                    # Total number of nodes
#SBATCH -n 8                    # Total number of tasks
#SBATCH -t 24:00:00             # Run time (hh:mm:ss)
#SBATCH -A NAIRR240454          # Project allocation name

module load gcc cuda

source $WORK/miniconda3/bin/activate ddmoe
export WANDB_API_KEY=2b60f655a687ad1161d31f0002256865e1ace428

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=23333

export OUTPUT_DIR=$WORK/outputs
export HF_HOME=$WORK/hf_cache
mkdir -p $OUTPUT_DIR HF_HOME


export SUB_TAS
srun --ntasks=$SLURM_NTASKS accelerate launch --config_file configs/slurm-8gpu.yaml \
  --num_processes=$SLURM_NTASKS \
  --num_machines=1 \
  --machine_rank=0 \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  --mixed_precision=fp16 \
  $WORK/scripts/finetune-sft.py \
  --output_dir=$OUTPUT_DIR \
  --base_model_name="allenai/OLMoE-1B-7B-0125" \
  --output_dir="outputs/olmoe-sft" \
  --dataset_name="Phando/OpenThoughts-114k-SFT" \
  --dataset_filter_condition="example['source'] == 'riddle_sense'"
