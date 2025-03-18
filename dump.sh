export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=0,1
#python scripts/dump-hidden-states.py \
#    --checkpoint_path="checkpoints/olmoe-1b-7b-0125-sft-distilled-moonlight/checkpoint-4000" --content="router_tokens" \
#    --save_dir="./outputs/profiling/olmoe-distill-4000"
#python scripts/dump-hidden-states.py \
#    --checkpoint_path="allenai/OLMoE-1B-7B-0125" --content="router_tokens" \
#    --save_dir="./outputs/profiling/olmoe-original"
python scripts/dump-hidden-states.py \
    --checkpoint_path="moonshotai/Moonlight-16B-A3B-Instruct" --content="router_tokens" \
    --save_dir="./outputs/profiling/moonlight"
#python scripts/dump-hidden-states.py \
#  --checkpoint_path="moonshotai/Moonlight-16B-A3B-Instruct" --content="router_tokens" \
#  --save_dir="./outputs/profiling/moonlight"
#python scripts/profile-routing.py \
#    --all_checkpoint_paths="moonshotai/Moonlight-16B-A3B-Instruct,allenai/OLMoE-1B-7B-0125,checkpoints/olmoe-1b-7b-0125-sft-distilled-moonlight/checkpoint-4000"