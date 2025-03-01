export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=0,1,2,3
#python debug.py
#python scripts/generate-distillation-data.py
accelerate launch --num_processes=4 scripts/generate-distillation-data.py