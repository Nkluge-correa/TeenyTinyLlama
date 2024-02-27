import sys
import math
import logging
import argparse
import datasets
import transformers
from pathlib import Path
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator)

from datasets import load_dataset
from codecarbon import EmissionsTracker
from accelerate.logging import get_logger
from accelerate import Accelerator, DistributedType

# These environment variables result in improved performance in modern Ampere GPUs (e.g., A100)
# Remember that `TF32` mode will only work on Ampere GPUs! 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main(args):
    
    # We are going to be using the `accelerate` library, which provides the `Accelerator` class
    # that can be used to handle device placement and distributed training.
    accelerator = Accelerator()

    # Set the logger.
    logger = get_logger(args.logger_name)

    # Create configurations for the logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # We are setting the verbosity of the `datasets`, `transformers` and `huggingface_hub` libraries
    # to `error` to avoid unnecessary logs.
    logger.info(accelerator.state, main_process_only=False)
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    huggingface_hub.utils.logging.set_verbosity_error()

    # We are using the `accelerator.wait_for_everyone()` method to ensure that all processes
    # have finished the previous steps before moving on to the next one.
    accelerator.wait_for_everyone()

    # Load the model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint_path, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint_path, revision=args.revision,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16 if args.attn_implementation else torch.float32,
    )

    # Load the evaluation dataset.
    #
    # The dataset folder must contain a list o parquet files, and you 
    # can achieve this by simply cloning the dataset from the hub 
    # to a local directory:
    #
    # git lfs install
    # git clone https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-tokenized
    #
    # Then, you should separate the dataset into `train` and `test` folders.
    # You will save time by having the dataset already tokenized and saved in a local folder.
    eval_dataset = load_dataset(
            'parquet',
            data_files={
                "test": f'{args.eval_folder_path}/test/*.parquet',
            },
            streaming=False)['test']

    # Set the format to `torch`.
    eval_dataset = eval_dataset.with_format("torch") 
    
    # Create the Evaluation DataLoader.
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        collate_fn=default_data_collator, 
        batch_size=args.per_device_eval_batch_size,
        pin_memory=True,
    )
    
    # We are preparing everything with `accelerator`. The `prepare` method will handle the device 
    # placement and distributed training.
    model, eval_dataloader = accelerator.prepare(
            model, eval_dataloader
        )
        
    # Create the an instance of `EmissionsTracker`.
    tracker = EmissionsTracker(
        log_level="critical", # set to "critical" to silence codecarbon
        output_file=f"emissions.csv",
        tracking_mode='machine',
    )

    logger.info(f"Running evaluation at step {args.completed_steps}.")

    model.eval()
    losses = []

    tracker.start()
    for step, batch in enumerate(tqdm(eval_dataloader, total=len(eval_dataloader), position=0, leave=True, disable=not accelerator.is_local_main_process, unit=" samples",  desc="Validation")):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
    
    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        eval_loss = torch.mean(losses)
        perplexity = float("inf")
    
    logger.info(f"Step {args.completed_steps} | Perplexity: {perplexity} | Evaluation Loss: {eval_loss} | Total Energy Consumption: {tracker._total_energy.kWh + args.total_energy_consumption}")
    
    # Print the results as a markdown table.
    print("| Step | Evaluation Loss | Perplexity | Total Energy Consumption |")
    print("| ---- | --------------- | ---------- |------------------------- |")
    print(f"| {args.completed_steps} | {eval_loss} | {perplexity} | {tracker._total_energy.kWh + args.total_energy_consumption} |")
    
    tracker.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on a checkpoint.")
    parser.add_argument("--logger_name", type=str, default="TeenyTinyLlama", help="Name of the logger.")
    parser.add_argument("--model_checkpoint_path", type=str, default="TeenyTinyLlama-460m", help="Path to the model checkpoint.")
    parser.add_argument("--revision", type=str, default="step100000", help="Revision of the model checkpoint.")
    parser.add_argument("--attn_implementation", type=str, default=None, help="What Attention implementation to use.")
    parser.add_argument("--eval_folder_path", type=str, default="/data/eval", help="Path to the evaluation folder.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument("--completed_steps", type=int, default=100000, help="Number of steps completed.")
    parser.add_argument("--total_energy_consumption", type=float, default=9.14, help="Total energy consumption until the checkpoint.")
    args = parser.parse_args()
    main(args)

# How to run:
# python evaluation.py --logger_name "TeenyTinyLlama" --model_checkpoint_path "nicholasKluge/TeenyTinyLlama-460m" --revision "step100000" --attn_implementation "flash_attention_2" --eval_folder_path "/data/Pt-Corpus-Instruct-tokenized-large" --per_device_eval_batch_size 16 --completed_steps 100000 --total_energy_consumption 3.34
