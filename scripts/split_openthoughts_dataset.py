#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script splits the 'open-thoughts/OpenThoughts-114k' dataset into two versions:
1. "OpenThoughts-114k-SFT" - Contains all columns except 'deepseek_solution'
2. "OpenThoughts-114k-R1-Distill" - Contains all columns except 'ground_truth_solution'

Then uploads both to the Hugging Face Hub under:
- "Phando/OpenThoughts-114k-SFT"
- "Phando/OpenThoughts-114k-R1-Distill"
"""

import os
import logging
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, login
from fire import Fire

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_and_push_datasets(
    source_dataset: str = "open-thoughts/OpenThoughts-114k",
    subset: str = "metadata",
    hf_token: str = None,
    use_auth_token: bool = False
):
    """
    Split the OpenThoughts dataset and push to HF Hub.
    
    Args:
        source_dataset: The source dataset on Hugging Face Hub
        subset: The subset of the source dataset to use (as positional parameter)
        hf_token: HuggingFace API token (alternative to use_auth_token)
        use_auth_token: Whether to use HF_TOKEN from environment
    """
    # Authenticate with HuggingFace
    if hf_token:
        login(token=hf_token)
    elif use_auth_token:
        if "HF_TOKEN" not in os.environ:
            raise ValueError("use_auth_token=True but HF_TOKEN not found in environment")
        login(token=os.environ["HF_TOKEN"])
    else:
        logger.warning("No authentication provided. You may hit rate limits or be unable to push datasets.")
    
    # Load the source dataset - note that subset is a positional parameter, not a named one
    logger.info(f"Loading dataset: {source_dataset}, subset: {subset}")
    try:
        # Load with subset as positional parameter
        dataset = load_dataset(source_dataset, subset, trust_remote_code=True)
        logger.info(f"Successfully loaded dataset with subset: {subset}")
    except Exception as e:
        logger.error(f"Error loading dataset with subset: {e}")
        logger.info("Trying to load dataset without subset...")
        dataset = load_dataset(source_dataset, trust_remote_code=True)
    
    # Log dataset info
    logger.info(f"Dataset loaded: {dataset}")
    logger.info(f"Dataset features: {dataset['train'].features}")
    logger.info(f"Number of examples: {len(dataset['train'])}")
    
    # Create the two variants
    logger.info("Creating SFT dataset (without deepseek_solution)")
    sft_dataset = Dataset.from_dict(
        {
            key: dataset['train'][key] 
            for key in dataset['train'].features 
            if key != 'deepseek_solution'
        }
    )
    
    logger.info("Creating R1-Distill dataset (without ground_truth_solution)")
    distill_dataset = Dataset.from_dict(
        {
            key: dataset['train'][key] 
            for key in dataset['train'].features 
            if key != 'ground_truth_solution'
        }
    )
    
    # Log sizes of the new datasets
    logger.info(f"SFT dataset size: {len(sft_dataset)}")
    logger.info(f"SFT dataset features: {list(sft_dataset.features.keys())}")
    logger.info(f"R1-Distill dataset size: {len(distill_dataset)}")
    logger.info(f"R1-Distill dataset features: {list(distill_dataset.features.keys())}")
    
    # Push to HuggingFace Hub
    logger.info("Pushing datasets to HuggingFace Hub...")
    
    sft_dataset.push_to_hub(
        "Phando/OpenThoughts-114k-SFT",
        private=False,
        commit_message="Upload OpenThoughts-114k SFT dataset (without deepseek_solution)"
    )
    logger.info("Successfully pushed SFT dataset to Phando/OpenThoughts-114k-SFT")
    
    distill_dataset.push_to_hub(
        "Phando/OpenThoughts-114k-R1-Distill",
        private=False,
        commit_message="Upload OpenThoughts-114k R1-Distill dataset (without ground_truth_solution)"
    )
    logger.info("Successfully pushed R1-Distill dataset to Phando/OpenThoughts-114k-R1-Distill")
    
    logger.info("Dataset splitting and pushing completed successfully!")

if __name__ == "__main__":
    Fire(split_and_push_datasets) 