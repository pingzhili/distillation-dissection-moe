export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=3

# "apps" "camelai_biology" "camelai_chemistry" "camelai_physics" "code_contests" "codeforces" "numina_math" "riddle_sense" "taco"
for TASK in "code_contests" "codeforces" "numina_math" "riddle_sense" "taco";
do
   python scripts/dump-hidden-states.py \
      --checkpoint_path="./outputs/olmoe-distill/olmoe-distill-$TASK" --content="router_tokens" \
      --save_dir="./outputs/profiling/olmoe-distill-$TASK"
done