from ddmoe.evaluation import evaluate_predictions
import json
from loguru import logger

RESULTS_PATH = "outputs/llama-3.2-1b-distill--qwen-antidistill-coef0.00003-temp2-epoch2-lr5e-5-checkpoint-60/checkpoint-177/gsm8k-results.json"

with open(RESULTS_PATH, "r") as f:
    results = json.load(f)
    
predictions = [res["prediction"] for res in results["content"]]
references = [res["ground_truth"] for res in results["content"]]

results = evaluate_predictions(predictions, references, is_math_task=True, use_last_number=True)

logger.info(f"Results: {results}")
