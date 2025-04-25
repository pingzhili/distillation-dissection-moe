export PYTHONPATH=$PYTHONPATH:src
CUDA_VISIBLE_DEVICES=4 python scripts/measure.py --metrics=expert_specialization --model_source="sft"  