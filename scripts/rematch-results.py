import json
import os
from fire import Fire
from loguru import logger
from ddmoe.evaluation import evaluate_predictions


def rematch_results(results_path: str, is_math_task: bool = False, is_choice_task: bool = False):
    with open(results_path, "r") as f:
        results = json.load(f)

    content = results["content"]
    predictions = []
    references = []
    
    for item in content:
        predictions.append(item["prediction"])
        references.append(item["ground_truth"])

    results = evaluate_predictions(predictions, references, is_math_task=is_math_task, is_choice_task=is_choice_task)
    logger.info(f"Results: {results} for {results_path}")


if __name__ == "__main__":
    Fire(rematch_results)
