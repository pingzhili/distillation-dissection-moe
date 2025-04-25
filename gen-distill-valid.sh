export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=7

for SUB_TASK in "riddle_sense" "camelai_biology" "camelai_chemistry" "camelai_physics" "taco" "code_contests" "codeforces" "apps" "numina_math"; do
    python scripts/generate-valid-data.py \
        --model_name="outputs/olmoe-distill/olmoe-distill-$SUB_TASK" \
        --save_dir="outputs/olmoe-distill/olmoe-distill-$SUB_TASK/gen-valid.csv"
    
    python scripts/dump-hidden-states.py \
        --content="router_tokens" \
        --checkpoint_path="outputs/olmoe-distill/olmoe-distill-$SUB_TASK" \
        --save_dir="outputs/olmoe-distill/olmoe-distill-$SUB_TASK/" \
        --valid_dataset="outputs/olmoe-distill/olmoe-distill-$SUB_TASK/gen-valid.csv"
done
