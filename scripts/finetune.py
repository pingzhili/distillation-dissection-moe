import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from fire import Fire
from transformers import (
    AutoTokenizer,
    set_seed,
    AutoModelForCausalLM,
    get_scheduler
)
from ddmoe.data import batch_preprocess_fn, CustomDataCollatorWithPadding


