export PYTHONPATH=$PYTHONPATH:src

# python scripts/rematch-results.py \
# --results_path="outputs/meta-llama/Llama-3.2-1B-Instruct/arcc-results.json" \
# --is_math_task=False --is_choice_task=True

# python scripts/rematch-results.py \
# --results_path="outputs/llama-3.2-1b-distill--qwen-antidistill-coef0.00003-temp2-epoch2-lr5e-5-checkpoint-60/checkpoint-177/arcc-results.json" \
# --is_math_task=False --is_choice_task=True

# python scripts/rematch-results.py \
# --results_path="outputs/llama-3.2-1b-distill--qwen-antidistill-coef0.00003-temp2-epoch2-lr5e-5-checkpoint-60/checkpoint-177/csqa-results.json" \
# --is_math_task=False --is_choice_task=True

# python scripts/rematch-results.py \
# --results_path="outputs/llama-3.2-1b-distill--qwen-antidistill-coef0.00003-temp2-epoch2-lr5e-5-checkpoint-60/checkpoint-177/math-results.json" \
# --is_math_task=True --is_choice_task=False

# python scripts/rematch-results.py \
# --results_path="outputs/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/arcc-results.json" \
# --is_math_task=False --is_choice_task=True

# python scripts/rematch-results.py \
# --results_path="outputs/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/csqa-results.json" \
# --is_math_task=False --is_choice_task=True

python scripts/rematch-results.py \
--results_path="outputs/llama-3.2-1b-distill--qwen-antidistill-coef0.00003-temp2-epoch2-lr5e-5-checkpoint-60/checkpoint-177/math-results.json" \
--is_math_task=True --is_choice_task=False