export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=0

# "apps" "camelai_biology" "camelai_chemistry" "camelai_physics" "code_contests" "codeforces" "numina_math" "riddle_sense" "taco"
for TASK in "apps" "camelai_biology" "camelai_chemistry" "camelai_physics";
do
   python scripts/dump-hidden-states.py \
      --checkpoint_path="./outputs/olmoe-sft/olmoe-sft-$TASK" --content="router_tokens" \
      --save_dir="./outputs/profiling/olmoe-sft-$TASK"
done