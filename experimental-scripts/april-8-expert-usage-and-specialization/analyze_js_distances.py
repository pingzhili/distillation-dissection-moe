#!/usr/bin/env python
import pickle
import numpy as np

# Load the data
data_path = 'experimental-scripts/april-8-expert-usage-and-specialization/outputs/expert_specialization_per_layer.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

# Collect JS distances for each layer
js_b = []  # Model B to A distances
js_c = []  # Model C to A distances
layer_names = []

for key in data.keys():
    if key == 'all_tasks':
        continue
        
    if 'distances_B_to_A' in data[key] and 'distances_C_to_A' in data[key]:
        b_dist = data[key]['distances_B_to_A'][~np.isinf(data[key]['distances_B_to_A'])]
        c_dist = data[key]['distances_C_to_A'][~np.isinf(data[key]['distances_C_to_A'])]
        
        if len(b_dist) > 0 and len(c_dist) > 0:
            js_b.append(np.mean(b_dist))
            js_c.append(np.mean(c_dist))
            layer_names.append(key)

# Print summary statistics
print("Jensen-Shannon Distance Analysis by Layer")
print("=========================================")
print(f"Average JS distance, Model B to A: {np.mean(js_b):.6f}")
print(f"Average JS distance, Model C to A: {np.mean(js_c):.6f}")
improvement = (np.mean(js_b) - np.mean(js_c)) / np.mean(js_b) * 100
print(f"Average improvement: {improvement:.2f}%")
c_closer_count = sum(c < b for c, b in zip(js_c, js_b))
print(f"Layers where C is closer to A: {c_closer_count} out of {len(js_b)} layers ({c_closer_count/len(js_b)*100:.1f}%)")

# Print per-layer analysis
print("\nPer-Layer Analysis:")
print("------------------")
print(f"{'Layer':<10} {'B to A':<10} {'C to A':<10} {'Diff':<10} {'Improvement':<10} {'Closer'}")
print("-" * 65)

for i, (layer, b, c) in enumerate(zip(layer_names, js_b, js_c)):
    diff = b - c
    imp = (diff / b) * 100 if b > 0 else 0
    closer = "C" if c < b else "B"
    print(f"{layer:<10} {b:.6f} {c:.6f} {diff:.6f} {imp:>8.2f}% {closer}") 