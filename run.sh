export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# python debug.py
# python scripts/generate-distillation-data.py
accelerate launch --num_processes=8 scripts/generate-distillation-data.py