export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=4


# python scripts/generate-valid-data.py \
#     --model_name="outputs/moonlight-instruct-sft-r1-distill/checkpoint-1000" \
#     --save_dir="outputs/moonlight-instruct-sft-r1-distill/checkpoint-1000/gen-valid.csv"
    
python scripts/dump-hidden-states.py \
    --content="router_tokens" \
    --checkpoint_path="outputs/moonlight-instruct-sft-r1-distill/checkpoint-1000" \
    --save_dir="outputs/moonlight-instruct-sft-r1-distill/checkpoint-1000/" \
    --valid_dataset="outputs/moonlight-instruct-sft-r1-distill/checkpoint-1000/gen-valid.csv"
