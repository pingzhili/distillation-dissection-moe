export PYTHONPATH=$PYTHONPATH:src

for SUB_TASK in "riddle_sense" "camelai_biology" "camelai_chemistry" "camelai_physics" "taco" "code_contests" "codeforces" "apps" "numina_math"; do
    python scripts/run-rse-baseline.py \
        --model_csv_path="outputs/gen-valid/olmoe-distill-$SUB_TASK.csv" 
    python scripts/run-rse-baseline.py \
        --model_csv_path="outputs/gen-valid/olmoe-sft-$SUB_TASK.csv"
done
