import pickle

DISTILL_SIMILRITY_PATH = "outputs/profiling/measure/distill_expert_specialization.pkl"
SFT_SIMILRITY_PATH = "outputs/profiling/measure/sft_expert_specialization.pkl"
with open(DISTILL_SIMILRITY_PATH, "rb") as f:
    distill_similarity_data = pickle.load(f)

with open(SFT_SIMILRITY_PATH, "rb") as f:
    sft_similarity_data = pickle.load(f)

task_to_per_layer_results = {}

for task, sim_dict in distill_similarity_data.items():
    per_layer_results = [None] * 15
    for layer, layer_sim in sim_dict.items():
        if layer == "all_tasks":
            continue
        distill_mean_distances = layer_sim["min_distances"].mean()
        sft_mean_distances = sft_similarity_data[task.replace("distill", "sft")][layer]["min_distances"].mean()
        layer = layer.split("_")[-1]
        per_layer_results[int(layer)-1] = (distill_mean_distances, sft_mean_distances, distill_mean_distances - sft_mean_distances)
    task_to_per_layer_results[task] = per_layer_results

for task, per_layer_results in task_to_per_layer_results.items():
    print(task, per_layer_results[0])
        
