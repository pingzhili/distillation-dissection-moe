export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=4

# for SUB_TASK in "riddle_sense" "camelai_biology" "camelai_chemistry" "camelai_physics" "taco" "code_contests" "codeforces" "apps" "numina_math"; do
#     python scripts/generate-valid-data.py \
#         --model_name="outputs/olmoe-sft/olmoe-sft-$SUB_TASK" \
#         --save_dir="outputs/olmoe-sft/olmoe-sft-$SUB_TASK/gen-valid.csv"
    
#     python scripts/dump-hidden-states.py \
#         --content="router_tokens" \
#         --checkpoint_path="outputs/olmoe-sft/olmoe-sft-$SUB_TASK" \
#         --save_dir="outputs/olmoe-sft/olmoe-sft-$SUB_TASK/" \
#         --valid_dataset="outputs/olmoe-sft/olmoe-sft-$SUB_TASK/gen-valid.csv"
# done


python scripts/generate-valid-data.py \
    --model_name="outputs/moonlight-instruct-sft-r1-distill/checkpoint-1000" \
    --save_dir="outputs/moonlight-instruct-sft-r1-distill/checkpoint-1000/gen-valid.csv"
    
python scripts/dump-hidden-states.py \
    --content="router_tokens" \
    --checkpoint_path="outputs/moonlight-instruct-sft-r1-distill/checkpoint-1000" \
    --save_dir="outputs/moonlight-instruct-sft-r1-distill/checkpoint-1000/" \
    --valid_dataset="outputs/moonlight-instruct-sft-r1-distill/checkpoint-1000/gen-valid.csv"
