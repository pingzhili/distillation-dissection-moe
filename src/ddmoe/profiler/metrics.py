import numpy as np
import torch
from collections import defaultdict
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy import stats
from loguru import logger
from tqdm import tqdm

class RoutingSimilarityMeasure:
    def __init__(self, reference_model_hidden_states, target_model_hidden_states):
        """
        Parameters:
            reference_model_hidden_states: dumped hidden states of the reference model
            target_model_hidden_states: dumped hidden states of the target model
        """
        self.reference_model_hidden_states = reference_model_hidden_states
        self.target_model_hidden_states = target_model_hidden_states

    
    def _compute_expert_task_distributions(self, model_hidden_states):
        """
        Compute the distribution of tasks for each expert in each layer.
        
        Returns
        =======
        layer_expert_task_distributions: Dict[layer_name: List[Dict[task_name: float]]]
            A dictionary of layer names to expert task distributions
        all_tasks: List[task_name]
            A list of all tasks
        """
        sources = model_hidden_states['sources']
        
        # Group by source/task
        task_groups = {}
        for i, source in enumerate(sources):
            task_name = source.split('/')[0] if '/' in source else source
            if task_name not in task_groups:
                task_groups[task_name] = []
            task_groups[task_name].append(i)
    
        # Extract model layers with MoE modules
        moe_layers = [key for key in model_hidden_states.keys() if 'model.layers' in key and '.mlp' in key]
        
        # Determine number of experts
        first_layer = moe_layers[0]
        num_experts = model_hidden_states[first_layer]['selected_experts'][0].max().item() + 1
        
        layer_expert_task_distributions = {}
    
        for layer in moe_layers:
            layer_name = layer.replace('model.layers.', 'layer_').replace('.mlp', '')
        
            # Initialize task counts for each expert in this layer
            expert_task_counts = [defaultdict(int) for _ in range(num_experts)]
            expert_total_counts = np.zeros(num_experts)
        
            # Count task occurrences for each expert in this layer
            for task_name, sample_indices in task_groups.items():
                for idx in sample_indices:
                    selected_experts = model_hidden_states[layer]['selected_experts'][idx]
                    for experts_per_token in selected_experts:
                        for expert_idx in experts_per_token:
                            expert_task_counts[expert_idx.item()][task_name] += 1
                            expert_total_counts[expert_idx.item()] += 1
        
            # Convert to probability distributions
            expert_task_distributions = []
            for expert_idx in range(num_experts):
                if expert_total_counts[expert_idx] > 0:
                    dist = {task: count / expert_total_counts[expert_idx] 
                        for task, count in expert_task_counts[expert_idx].items()}
                else:
                    dist = {}
                expert_task_distributions.append(dist)
        
            layer_expert_task_distributions[layer_name] = expert_task_distributions
    
        return layer_expert_task_distributions, list(task_groups.keys())
    
    
    def _compute_expert_entropies(self, expert_task_distributions, all_tasks):
        """Compute entropy of task distribution for each expert."""
        expert_entropies = []
        
        for expert_dist in expert_task_distributions:
            # Convert to array with zeros for missing tasks
            dist_array = np.zeros(len(all_tasks))
            for i, task in enumerate(all_tasks):
                dist_array[i] = expert_dist.get(task, 0)
        
            # Skip if expert is not used
            if np.sum(dist_array) == 0:
                expert_entropies.append(0)
                continue
        
            # Normalize if needed
            dist_array = dist_array / np.sum(dist_array)
            
            # Compute entropy
            expert_entropies.append(entropy(dist_array))
        
        return np.array(expert_entropies)
    
    def _find_closest_experts(self, expert_dist_A, expert_dist_B, all_tasks):
        """Find closest expert in B for each expert in A using Jensen-Shannon divergence."""
        num_experts_A = len(expert_dist_A)
        num_experts_B = len(expert_dist_B)
        
        # Initialize distance matrix
        distances = np.zeros((num_experts_A, num_experts_B))
        
        # Compute JS divergence between each pair of experts
        for i in range(num_experts_A):
            dist_A = np.zeros(len(all_tasks))
            for j, task in enumerate(all_tasks):
                dist_A[j] = expert_dist_A[i].get(task, 0)
            
            # Skip if expert is not used
            if np.sum(dist_A) == 0:
                distances[i, :] = np.inf
                continue
            
            # Normalize
            dist_A = dist_A / np.sum(dist_A)
            
            for j in range(num_experts_B):
                dist_B = np.zeros(len(all_tasks))
                for k, task in enumerate(all_tasks):
                    dist_B[k] = expert_dist_B[j].get(task, 0)
                
                # Skip if expert is not used
                if np.sum(dist_B) == 0:
                    distances[i, j] = np.inf
                    continue
                
                # Normalize
                dist_B = dist_B / np.sum(dist_B)
                
                # Compute Jensen-Shannon divergence
                distances[i, j] = jensenshannon(dist_A, dist_B)
        
        # Find closest expert in B for each expert in A
        closest_experts = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)
        
        return closest_experts, min_distances
        
    def compute_expert_specialization_similarity(self):
        """
        Compute the similarity between the expert specialization of the reference and target models
        """
        ref_layer_dists, ref_tasks = self._compute_expert_task_distributions(self.reference_model_hidden_states)
        tgt_layer_dists, tgt_tasks = self._compute_expert_task_distributions(self.target_model_hidden_states)
        all_tasks = set(ref_tasks) & set(tgt_tasks)
        all_tasks = sorted(list(all_tasks))

        results = {'all_tasks': all_tasks}

        # Get all layer names from one model
        layer_names = list(ref_layer_dists.keys())

        for layer_name in layer_names:
            if layer_name not in tgt_layer_dists or layer_name not in ref_layer_dists:
                continue
            layer_results = {}

            # Compute expert entropies for each model for this layer
            ref_expert_entropies = self._compute_expert_entropies(ref_layer_dists[layer_name], all_tasks)
            tgt_expert_entropies = self._compute_expert_entropies(tgt_layer_dists[layer_name], all_tasks)
            
            # Find closest experts from tgt to ref
            # breakpoint()
            closest_experts, min_distances = self._find_closest_experts(
                ref_layer_dists[layer_name],
                tgt_layer_dists[layer_name],
                all_tasks
            )

            layer_results['ref_expert_entropies'] = ref_expert_entropies
            layer_results['tgt_expert_entropies'] = tgt_expert_entropies
            layer_results['closest_experts'] = closest_experts
            layer_results['min_distances'] = min_distances

            results[layer_name] = layer_results
        
        return results
    
    def compute_expert_specialzation_entropy_similarity(self):
        ref_layer_dists, ref_tasks = self._compute_expert_task_distributions(self.reference_model_hidden_states)
        tgt_layer_dists, tgt_tasks = self._compute_expert_task_distributions(self.target_model_hidden_states)
        all_tasks = set(ref_tasks) & set(tgt_tasks)
        all_tasks = sorted(list(all_tasks))



