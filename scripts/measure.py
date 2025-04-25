from loguru import logger
from src.ddmoe.profiler.metrics import RoutingSimilarityMeasure
import os
from fire import Fire
import torch
from tqdm import tqdm
import pickle
# REFERENCE_MODEL_HIDDEN_STATES = "outputs/profiling/moonlight-instruct-sft-r1-distill-1000/router_tokens.pt"
REFERENCE_MODEL_HIDDEN_STATES = "outputs/moonlight-instruct-sft-r1-distill/checkpoint-1000/router_tokens.pt"
TARGET_MODEL_HIDDEN_STATES_DICT = {}


def measure(metrics: str, model_source: str="sft"):
    for file in os.listdir(f"outputs/olmoe-{model_source}"):
        if f"olmoe-{model_source}-" in file:
            TARGET_MODEL_HIDDEN_STATES_DICT[file] = os.path.join(f"outputs/olmoe-{model_source}", file, "router_tokens.pt")

    logger.info(f"Loading target models: {TARGET_MODEL_HIDDEN_STATES_DICT.keys()}")
    logger.info(f"Loading reference model: {REFERENCE_MODEL_HIDDEN_STATES}")

    measure_results = {}
    for target_model_name, target_model_hidden_states in tqdm(TARGET_MODEL_HIDDEN_STATES_DICT.items(), desc="Measuring target models"):
        logger.info(f"Measuring {target_model_name}")
        measurer = RoutingSimilarityMeasure(
            reference_model_hidden_states=torch.load(REFERENCE_MODEL_HIDDEN_STATES),
            target_model_hidden_states=torch.load(target_model_hidden_states)
        )
        if metrics == "expert_specialization":
            results = measurer.compute_expert_specialization_similarity()
        else:
            raise ValueError(f"Invalid metrics: {metrics}")
        measure_results[target_model_name] = results
    
    # save results
    with open(f"outputs/profiling/{model_source}_measure_results_{metrics}.pkl", "wb") as f:
        pickle.dump(measure_results, f)

    return measure_results

if __name__ == "__main__":
    Fire(measure)

# # pickle load
# with open("outputs/profiling/distill_measure_results_expert_specialization.pkl", "rb") as f:
#     sft_measure_results = pickle.load(f)
#     for tgt_model_name, tgt_model_results in sft_measure_results.items():
#         all_layer_min_distances_results = []
#         for layer_name, layer_results in tgt_model_results.items():
#             if "layer_" not in layer_name:
#                 continue
#             all_layer_min_distances_results.append(layer_results["min_distances"].mean().item())
#         print(tgt_model_name, sum(all_layer_min_distances_results) / len(all_layer_min_distances_results))


