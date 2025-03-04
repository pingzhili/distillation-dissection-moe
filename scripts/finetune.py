import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import transformers
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
    AutoModelForCausalLM,
    get_scheduler,
    DataCollatorWithPadding
)
from ddmoe.data import batch_preprocess_fn, CustomDataCollatorWithPadding

set_seed(233)
logger = get_logger(__name__)


def train_sft(
        base_model_name: str = "allenai/OLMoE-1B-7B-0125",
        dataset_name: str = "Phando/sft-dataset-from-moonlight",
        max_length: int = 1024,
        batch_size_per_device: int = 16,
        gradient_accumulation_steps: int = 1,
        num_train_epochs: int = 3,
        learning_rate: float = 5e-6,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        output_dir: str = "./outputs/",
        num_workers: int = 4,
        checkpointing_steps: int = 1000,
        logging_steps: int = 1,
):
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps, project_dir=output_dir, log_with="wandb"
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    raw_datasets = load_dataset(dataset_name, split="train", trust_remote_code=True)

    # debugging
    raw_datasets = raw_datasets.select(range(1000))

    if "olmoe" in base_model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125-Instruct", trust_remote_code=True)
    else:
        raise NotImplementedError(f"Tokenizer for {base_model_name} not implemented.")
    tokenizer.model_max_length = max_length
    model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)

    with accelerator.main_process_first():
        columns_names = raw_datasets.column_names
        sft_dataset = raw_datasets.map(
            lambda x: batch_preprocess_fn(x, task="sft-olmoe-train", tokenizer=tokenizer),
            batched=True,
            remove_columns=columns_names,
            num_proc=num_workers,
        )
        print(sft_dataset[0])
    data_collator = CustomDataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8, max_length=max_length)
    dataloader = DataLoader(
        sft_dataset,
        collate_fn=data_collator,
        batch_size=batch_size_per_device,
        num_workers=num_workers,
        shuffle=True
    )
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    num_training_steps = num_update_steps_per_epoch * num_train_epochs
    warmup_steps = int(warmup_ratio * num_training_steps)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_train_epochs * len(dataloader),
    )
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    num_training_steps = num_update_steps_per_epoch * num_train_epochs
    logger.info(f"num_training_steps = {num_training_steps}")

    accelerator.init_trackers(project_name="ddmoe")

    # Train!
    total_batch_size = batch_size_per_device * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"total_batch_size = {total_batch_size}")
    logger.info(f"gradient_accumulation_steps = {gradient_accumulation_steps}")
    logger.info(f"num_train_epochs = {num_train_epochs}")
    logger.info(f"learning_rate = {learning_rate}")
    logger.info(f"weight_decay = {weight_decay}")
    logger.info(f"warmup_ratio = {warmup_ratio}")
    logger.info(f"output_dir = {output_dir}")
    logger.info(f"num_workers = {num_workers}")
    logger.info(f"checkpointing_steps = {checkpointing_steps}")
    logger.info(f"logging_steps = {logging_steps}")

    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    model.train()
    completed_steps = 0

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(dataloader):
            model.train()
            with accelerator.accumulate(model):
                # if accelerator.is_local_main_process:
                #     print(batch)
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                checkpointing_dir = os.path.join(output_dir, f"checkpoint-{completed_steps}")
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    checkpointing_dir, save_function=accelerator.save, is_main_process=accelerator.is_main_process
                )

            if completed_steps > num_training_steps:
                break

            accelerator.log({
                "loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
            })

    accelerator.end_training()
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    checkpointing_dir = os.path.join(output_dir, f"checkpoint-{completed_steps}")
    unwrapped_model.save_pretrained(
        checkpointing_dir, save_function=accelerator.save, is_main_process=accelerator.is_main_process
    )


if __name__ == "__main__":
    Fire(train_sft)
