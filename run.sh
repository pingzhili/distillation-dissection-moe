export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=0,1
#python debug.py
python scripts/generate-distillation-data.py