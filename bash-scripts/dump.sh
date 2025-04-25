export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=0,1,2,3

python scripts/dump-hidden-states.py \
      --checkpoint_path="./outputs/moonlight-instruct-sft-r1-distill/checkpoint-1000" --content="router_tokens" \
      --save_dir="./outputs/profiling/moonlight-instruct-sft-r1-distill-1000"