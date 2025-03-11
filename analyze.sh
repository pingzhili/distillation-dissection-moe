export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=0

python scripts/analyze-hidden-states.py \
  --before_router_checkpoint_path="allenai/OLMoE-1B-7B-0125/expert_profiling/router_tokens.pt" \
  --after_router_checkpoint_path="checkpoints/olmoe-1b-7b-0125-sft-distilled-moonlight/checkpoint-4000/expert_profiling/router_tokens.pt"