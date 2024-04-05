# Requirements to run this script:
# - Python version: 3.10.12
# - transformers==4.38.0.dev0
# - torch==2.1.0+cu121
# - pyyaml==6.0.1
# - datasets==2.16.1
# - wandb==0.16.2
# - codecarbon==2.3.3
# - huggingface_hub==0.20.2
# - accelerate==0.26.1
# - sentencepiece==0.1.99
# - flash-attn==2.5.0
# - deepspeed==0.14.0
#
# To run this script, in a single GPU, you just need to run:
#
# python pre-training.py --spec-file specs.yaml
#
# For distributed training, run this script using `accelerate`:
#
# accelerate launch --config_file "fsdp_config.yaml" --num_processes=4 pre-training.py --spec-file specs.yaml
#
# This will launch 4 processes on the current node, each with 1 GPU device per process.
# More information can be found here: https://huggingface.co/docs/accelerate/basic_tutorials/launch
# If the `accelerate lunch` is slower than just running the script without `accelerate`, this is probably due
# to some incompatibility with the current version of torch and the cuda driver. Issue documented here:
# https://discuss.huggingface.co/t/single-gpu-is-faster-than-multiple-gpus/71383
#
# The `fsdp_config.yaml` file is a configuration file for the `accelerate` library. It is used to set the
# number of processes, the number of GPUs per process, and other settings. You can get more information on the
# `accelerate` documentation and this step-by-step tutorial from Hugging Face on the links below.
#
# Documentationon Accelerate Config: https://huggingface.co/docs/accelerate/basic_tutorials/launch#why-you-should-always-use-accelerate-config
# Step-by-step Tutorial: https://huggingface.co/blog/ram-efficient-pytorch-fsdp

import os
import sys
import time
import yaml
import math
import json
import random
import logging
import warnings
import argparse
from tqdm.auto import tqdm
from pathlib import Path

import torch
import wandb
import datasets
import transformers
import huggingface_hub
from datasets import load_dataset
from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker
from huggingface_hub import create_repo, HfApi, create_branch

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TrainingArguments,
    default_data_collator,
    get_scheduler,
)

from specifications import ModelArguments, DataTrainingArguments, ExtraArguments

# These environment variables result in improved performance in modern Ampere GPUs (e.g., A100)
# Remember that `TF32` mode will only work on Ampere GPUs!
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main(spec_file):

    # Load the arguments from the spec file.
    with open(spec_file, "r") as stream:
        all_kwargs = yaml.safe_load(stream)

    # Separete the arguments for the model, data, training, and extra arguments (wandb, accelerate, etc.).
    # You can check the `specifications.py` file to see the structure of the arguments.
    model_args = ModelArguments(**all_kwargs['model_args'])
    data_args = DataTrainingArguments(**all_kwargs['data_args'])
    training_args = TrainingArguments(**all_kwargs['training_args'])
    extra_args = ExtraArguments(**all_kwargs['extra_args'])

    # We are going to be using the `accelerate` library, which provides the `Accelerator` class
    # that can be used to handle device placement and distributed training.
    accelerator = Accelerator(
        mixed_precision=extra_args.mixed_precision,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps, 
        project_dir=training_args.output_dir)

    # Create a directory to save the logs and the model checkpoints.
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Set the logger.
    # Nothing fancy here, just a simple logger.
    logger = get_logger(extra_args.logger_name)

    # Create configurations for the logger.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # We are setting the verbosity of the `datasets`, `transformers` and `huggingface_hub` libraries
    # to `error` to avoid unnecessary logs.
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    huggingface_hub.utils.logging.set_verbosity_error()

    # Log the status of the accelerator on all processes.
    logger.info(accelerator.state, main_process_only=False)

    # Set seed before initializing model.
    # This is important to ensure synchronization of the random number generators across all processes.
    if training_args.seed is not None:
        set_seed(training_args.seed)
    
    # We are using the `accelerator.wait_for_everyone()` method to ensure that all processes
    # have finished the previous steps before moving on to the next one.
    # Documentation: https://huggingface.co/docs/accelerate/v0.27.2/en/package_reference/accelerator#synchronicity-control       
    accelerator.wait_for_everyone()

    # Create a HuggingFace repository!
    # Ideally, we would like to create a HuggingFace repository for the model and the tokenizer,
    # but only the main process should do this. On the contrary, we will have multiple repositories
    # created by the non-main processes.
    if accelerator.is_main_process:
        if training_args.push_to_hub and training_args.hub_token is not None:
            if training_args.hub_model_id is not None:
                create_repo(
                    repo_id=f"{training_args.hub_model_id}", 
                    token=training_args.hub_token,
                    repo_type="model",
                    exist_ok=True,
                    private=True)
            
            else:
                raise ValueError("No model id provided. Try running with `hub_model_id=your-user-name/your-model-name`")

    # Now we are going to load the configuration, model and tokenizer from the HuggingFace Hub.
    # According to the documentation, the `from_pretrained` methods guarantee that only one local process can concurrently
    # download the model/tokenizer from the HuggingFace Hub.
    if model_args.tokenizer_name is not None:

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, 
            **{
            "cache_dir": model_args.cache_dir,
            "use_fast": model_args.use_fast_tokenizer,
            "revision": model_args.model_revision,
            "token": training_args.hub_token,
            "trust_remote_code": model_args.trust_remote_code,
            }
        )

        # Here we are going to load the tokenizer to our newly created HuggingFace repository.
        # If the `accelerator.is_main_process` is working as expected, only the main process should
        # be able to upload the tokenizer to the HuggingFace Hub.
        if accelerator.is_main_process:
            if training_args.push_to_hub and training_args.hub_token is not None:
                if training_args.hub_model_id is not None:
                    tokenizer.push_to_hub(training_args.hub_model_id, token=training_args.hub_token)
        
    else:
        raise ValueError("Need a tokenizer name to train on. Train a tokenizer from scratch usign the `train-sentencepiece.py`.")

    # See if we need to train the model from scratch.
    if model_args.train_from_scratch:

        logger.info("Training new model from scratch (train_from_stratch=True)")

        # Load the configurations to create a new model.
        configuration = AutoConfig.from_pretrained(
            model_args.model_to_train, 
            **{
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": training_args.hub_token,
            "trust_remote_code": model_args.trust_remote_code,
            "output_hidden_states": model_args.output_hidden_states,
            "hidden_size": model_args.hidden_size, # 160m: 768 | 460m: 1024 | 1.1b: 2048
            "intermediate_size": model_args.intermediate_size, # 160m: 3072 | 460m: 4096 | 1.1b: 5632
            "max_position_embeddings": model_args.max_position_embeddings, # 160m: 2048 | 460m: 2048 | 1.1b: 2048
            "num_attention_heads": model_args.num_attention_heads, # 160m: 12 | 460m: 16 | 1.1b: 32
            "num_hidden_layers": model_args.num_hidden_layers, # 160m: 12 | 460m: 24 | 1.1b: 22
            "num_key_value_heads": model_args.num_key_value_heads, # 160m: 12 | 460m: 16 | 1.1b: 4
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "torch_dtype": torch.bfloat16 if extra_args.mixed_precision.endswith('16') or model_args.attn_implementation else torch.float32,
            "vocab_size": len(tokenizer),
            "use_cache": model_args.use_cache,
            "attn_implementation": model_args.attn_implementation,
            }
        )

        # Create an instance of the model using the configuration.
        model = AutoModelForCausalLM.from_config(configuration)

        # Change the model's `name_or_path`.  
        model.config.name_or_path = training_args.hub_model_id

        # Resize the model's embedding layer to match the tokenizer's vocabulary size.
        model.resize_token_embeddings(len(tokenizer))

        # Count the number of trainable parameters in the model.
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'There are {params:,} trainable parameters in this model.')

        # Create generation config file (new configuration file recently added to the `transformers` library).
        generation_config = GenerationConfig(
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            unk_token_id=tokenizer.unk_token_id,
            max_length=model_args.max_position_embeddings,
        )

    else:

        logger.info("Fine-tuning model from HuggingFace Hub (train_from_stratch=False)")

        # Load the configuration of the model to train.
        configuration = AutoConfig.from_pretrained(
            model_args.model_to_train, 
            **{
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": training_args.hub_token,
            "trust_remote_code": model_args.trust_remote_code,
            "output_hidden_states": model_args.output_hidden_states,
            "torch_dtype": torch.bfloat16 if extra_args.mixed_precision.endswith('16') or model_args.attn_implementation else torch.float32,
            "attn_implementation": model_args.attn_implementation,
            }
        )

        # Load the pretrained model to fine-tune
        model = AutoModelForCausalLM.from_pretrained(
                model_args.model_to_train,
                config=configuration,
        )
        
        # Change the model's `name_or_path`. 
        model.config.name_or_path = training_args.hub_model_id

        # Resize the model's embedding layer to match the tokenizer's vocabulary size.
        model.resize_token_embeddings(len(tokenizer))

        # Count the number of trainable parameters in the model.
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'There are {params:,} trainable parameters in this model.')

        # Load the generation config file.
        generation_config = GenerationConfig.from_pretrained(model_args.model_to_train)

    # Gradient checkpointing can be enabled to reduce the memory usage during training.
    # However, this will slow down the training process by about 20%.
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Load the datasets.
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
    train_dataset = load_dataset(
            'parquet', 
            data_files={
                "train": f'{data_args.folder_path}/train/*.parquet',
            })['train']
    
    eval_dataset = load_dataset(
            'parquet',
            data_files={
                "test": f'{data_args.folder_path}/test/*.parquet',
            })['test']

    # Set the format to `torch`.
    train_dataset = train_dataset.with_format("torch")
    eval_dataset = eval_dataset.with_format("torch") 
    
    # If we wat to do a sanity check, we will use a small subset of the dataset.
    if data_args.sanity_check:

        logger.info(f"`Sanity check` is set to `True`. Train set size: 1000 | Validation set size: 100")

        train_dataset = train_dataset.select(range(1000)) 
        eval_dataset = eval_dataset.select(range(100))

    # Create the Training DataLoader and Evaluation DataLoader.
    if training_args.do_train and training_args.do_eval:
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=default_data_collator, 
            batch_size=training_args.per_device_train_batch_size,
            pin_memory=training_args.dataloader_pin_memory,
        )

        eval_dataloader = DataLoader(
            eval_dataset,
            shuffle=False,
            collate_fn=default_data_collator, 
            batch_size=training_args.per_device_eval_batch_size,
            pin_memory=training_args.dataloader_pin_memory,
        )
    
    # Create only the Training DataLoader.
    elif training_args.do_train and not training_args.do_eval:
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=default_data_collator, 
            batch_size=training_args.per_device_train_batch_size,
            pin_memory=training_args.dataloader_pin_memory,
        )

    # Now, we create our Optimizer. First, we will split weights in two groups, 
    # one with weight decay and the other not.
    # These strings `["bias", "layer_norm.weight"]` represent parameter names that should not be subject to weight decay during optimization. 
    # Weight decay is a regularization technique used during training to prevent overfitting by penalizing large weights.
    no_decay = ["bias", "layer_norm.weight"]

    # The first dictionary corresponds to parameters with weight decay (regularization) applied to them (non-bias and non-layer_norm.weight parameters).
    # The second dictionary corresponds to parameters without weight decay (regularization) applied to them (bias and layer_norm.weight parameters).
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # We are using the `AdamW` optimizer, which is a variant of the Adam optimizer.
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, 
        lr=training_args.learning_rate,
        eps=training_args.adam_epsilon,
    )
    
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)

    # Set number of max steps if they are not provided.
    if training_args.max_steps is None:
        training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        
    # Our scheduler will start with a warmup phase, where the learning rate will increase linearly from 0 to the initial learning rate
    # over the first n `num_warmup_steps` of the training steps. Then, the learning rate will decrease following the cosine function.
    # If the shape of the learning rate curve is not according to what we expect, there is something wrong with (probably) the `num_training_steps` parameter.
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps,
    )

    # We are preparing everything with `accelerator`. The `prepare` method will handle the device 
    # placement and distributed training.
    if training_args.do_train and training_args.do_eval:

        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    
    else:

        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )
    
    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Now, we need to recalculate our total training steps as the size of the training dataloader may have changed.
    # This change (I belive) should be due to the distributed training, where the dataset is split among the 
    # different processes.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    training_args.num_train_epochs = math.ceil(training_args.max_steps / num_update_steps_per_epoch)

    # Now, we want to log the training process to the Weights & Biases platform. We need to initialize the `wandb` 
    # logger and then log the training process to the platform. Since we are using distributed training, we 
    # need to ensure that only the main process logs the training process to the platform.
    if accelerator.is_main_process:
        
        if extra_args.wandb_token is not None: 

            # Login to wandb.
            wandb.login(key=extra_args.wandb_token)

            # Initialize wandb.
            wandb.init(
                project=extra_args.logger_name, 
                notes="Training a Llama 2 model on a custom Portuguese-BR dataset.",
                tags=["Energy Consumption", "Language Modeling", "Portuguese"],
                name=f"""{extra_args.logger_name.lower()}-{model_args.model_id}-{time.strftime("%d-%m-%Y")}""",
                config=all_kwargs,
                resume="allow",
                id=extra_args.logger_name + model_args.model_id,
            )

    # We would also like to track the energy consumption of the training process. We are going to use the `codecarbon` library
    # to do this. We need to initialize the `EmissionsTracker` and then track the energy consumption of the training process.
    tracker = EmissionsTracker(
        project_name=extra_args.logger_name,
        log_level="critical", # Set to "critical" to silence codecarbon.
        output_dir=training_args.output_dir,
        output_file=f"emissions_{accelerator.process_index}.csv",
        tracking_mode='machine', # We are tracking the energy consumption of the whole machine.
    )

    logger.info(f'Geo Location: ISO: {tracker._geo.country_iso_code} | Country: {tracker._geo.country_name} | Region : {tracker._geo.region}')

    # Initialize the HuggingFace Hub API.
    if training_args.push_to_hub and training_args.hub_token is not None:
        if training_args.hub_model_id is not None:
            api = HfApi(token=training_args.hub_token)

    # The total batch size is calculated by multiplying the number of samples in `per_device_train_batch_size`
    # by the number of processes in the accelerator.
    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}.")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs:.1f}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(training_args.max_steps), disable=not accelerator.is_local_main_process, unit=" samples", desc="Training")
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save.
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint is not None or training_args.resume_from_checkpoint != "":
            checkpoint_path = training_args.resume_from_checkpoint
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)
        
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)

        # Extract our `step_{i}`
        training_difference = os.path.splitext(path)[0]
        
        # Need to multiply `gradient_accumulation_steps` to reflect real steps.
        resume_step = int(training_difference.replace("step_", "")) * training_args.gradient_accumulation_steps
        starting_epoch = resume_step // len(train_dataloader)
        completed_steps = resume_step // training_args.gradient_accumulation_steps
        resume_step -= starting_epoch * len(train_dataloader)
    
    # Update the progress_bar.
    progress_bar.update(completed_steps)
    accelerator.print()

    # Start codecarbon tracking before we start the training loop.
    tracker.start()

    for epoch in range(starting_epoch, training_args.num_train_epochs):

        logger.info(f'Beginning epoch {epoch + 1} of {training_args.num_train_epochs}')

        # Set the model to training mode.
        model.train()
        total_loss = 0

        # Set the dataloader to the active dataloader.
        if training_args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint.
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            logger.info(f"Skipping the first {resume_step} batches of the first epoch.")
        else:
            active_dataloader = train_dataloader
        
        # Iterate over the batches of data in the current epoch.
        for step, batch in enumerate(active_dataloader, start=1):

            with accelerator.accumulate(model):
                
                # Forward pass the batch through the model and get the loss.
                outputs = model(**batch)
                loss = outputs.loss

                # Add the loss to the total loss.
                total_loss += loss.detach().float()

                # We only want to log the loss to wandb from the main process.
                if accelerator.is_main_process:
                    
                    if (step) % extra_args.wandb_log_steps == 0 and extra_args.wandb_token is not None:
                        wandb.log({
                            "loss": loss.detach().float().item(),
                            # Log the learning rate to wandb (this is how we can monitor the learning rate during training).
                            "lr": lr_scheduler.get_last_lr()[0],
                            })

                # Backward pass and update optimizer.
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update the progress bar. The `accelerator.sync_gradients` method is used to synchronize the gradients across all processes.
            # Hence, the progress bar is updated only when all processes have finished the current step.
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
            
            # Here, we are going to save the model checkpoint every `checkpointing_steps`.
            accelerator.wait_for_everyone()
                
            # Check if `checkpointing_steps` is an integer
            if isinstance(extra_args.checkpointing_steps, int):

                if completed_steps % extra_args.checkpointing_steps == 0 and completed_steps > 0:

                    output_dir = f"step_{completed_steps}"
                    if training_args.output_dir is not None:
                        # Join the output directory with the current checkpoint directory.
                        output_dir = os.path.join(training_args.output_dir, output_dir)
                    # Save the model checkpoint.
                    accelerator.save_state(output_dir)

                    # Save the generation config file in the checkpoint directory.
                    generation_config.save_pretrained(output_dir)

                    if accelerator.is_main_process:
                    
                        # Log the energy consumption to wandb.
                        if extra_args.wandb_token is not None:
                            wandb.log({
                                "total_energy_consumption": tracker._total_energy.kWh,      
                            })
                

                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )
                    tokenizer.save_pretrained(output_dir)

                    if accelerator.is_main_process:

                        # Push the model checkpoint to the hub.
                        if training_args.push_to_hub and training_args.hub_token is not None:
                            if training_args.hub_model_id is not None:

                                # Here we are going to push the model checkpoint to the HuggingFace Hub in a try-except block. 
                                # If the push to the Hub fails, we will log a warning.
                                try:

                                    logger.info(f"""Checkpoint directory (`{output_dir}`) being uploaded to the hub.""")

                                    # We are pushing every checkpoint to the Hub as a separate directory. Pushing different checkpoints to the same repo id as
                                    # different branches is not working as expected.
                        
                                    create_repo(
                                        repo_id=f"{training_args.hub_model_id}-step-{completed_steps}", 
                                        token=training_args.hub_token,
                                        repo_type="model",
                                        exist_ok=True,
                                        private=True
                                    )

                                    api.upload_folder(
                                        repo_id=f"{training_args.hub_model_id}-step-{completed_steps}",
                                        folder_path=output_dir,
                                    )

                                    logger.info(f"Checkpoint pushed to the hub at step {completed_steps}!")
                                
                                except Exception as e:
                                    logger.warning(f"Error while uploading checkpoint to Hub: {e}")
                    
                    # Upload to the Hub all the emmisions files created by the different processes.
                    tracker.flush()

                    if training_args.push_to_hub and training_args.hub_token is not None:
                        if training_args.hub_model_id is not None:

                            try:

                                api.upload_file(
                                    path_or_fileobj=f"{training_args.output_dir}/emissions_{accelerator.process_index}.csv",
                                    path_in_repo=f"emissions_{accelerator.process_index}.csv",
                                    repo_id=f"{training_args.hub_model_id}-step-{completed_steps}"
                                )

                                logger.info(f"Emissions file pushed to the hub at step {completed_steps}!")
                            
                            except Exception as e:
                                logger.warning(f"Error while uploading emissions file to Hub: {e}")
                                     
            # Generate text from the model every `sample_every ` steps.
            # The main process will generate the samples. Hence, we are passing the tensors to the GPU at
            # the main process only, which should be the process with the rank 0.                     
            if accelerator.is_main_process:
                
                if completed_steps % extra_args.sample_every == 0 and not completed_steps == 0:
                    
                    model.config.use_cache = True

                    try:

                        model.eval()
                            
                        # Sample a string from the `generation_seeds` and generate text from the model.
                        inputs = tokenizer(random.choice(extra_args.generation_seeds), return_tensors="pt").to('cuda:0')

                        sample_outputs = model.generate(**inputs,
                                            do_sample=True,
                                            top_k=50,
                                            max_length=150,
                                            repetition_penalty=1.2,
                                            top_p=0.50,
                                            num_return_sequences=5)
                        
                        model.config.use_cache = False
                        
                        texts = []

                        for i, sample_output in enumerate(sample_outputs):
                            texts.append(tokenizer.decode(sample_output))
                        
                        # Log the samples to the main process terminal.
                        for text in texts:
                            logger.info(f"Samples (Epoch: {epoch + 1} | Step: {completed_steps}): {text}")
                        
                        # Log the samples to wandb.
                        if extra_args.wandb_token is not None:

                            training_samples = wandb.Table(columns=[f"Samples (Epoch: {epoch + 1} | Step: {completed_steps})"])
                            for text in texts:
                                training_samples.add_data(text)
                            wandb.log({f"Samples (Epoch: {epoch + 1} | Step: {completed_steps})": training_samples})
                    
                    except Exception as e:
                        logger.warning(f"Error while generating samples: {e}")
                        model.config.use_cache = False

                    model.train()
            
            # Evaluation should be run on all processes.
            if training_args.do_eval:

                # Check if `evaluation_strategy=steps`
                if training_args.evaluation_strategy == "steps":

                    if completed_steps % training_args.eval_steps == 0 and completed_steps > 0:

                        accelerator.print()
                        logger.info(f"Running evaluation at step {completed_steps}.")

                        model.eval()
                        losses = []
                        for step, batch in enumerate(tqdm(eval_dataloader, total=len(eval_dataloader), position=0, leave=True, disable=not accelerator.is_local_main_process, unit=" samples",  desc="Validation")):
                            with torch.no_grad():
                                outputs = model(**batch)
                            loss = outputs.loss
                            losses.append(accelerator.gather_for_metrics(loss.repeat(training_args.per_device_eval_batch_size)))
                        
                        losses = torch.cat(losses)
                        try:
                            eval_loss = torch.mean(losses)
                            perplexity = math.exp(eval_loss)
                        except OverflowError:
                            eval_loss = torch.mean(losses)
                            perplexity = float("inf")
                        
                        logger.info(f"Step {completed_steps} | Perplexity: {perplexity} | Average Training Loss: {total_loss.item() / completed_steps} | Evaluation Loss: {eval_loss} | Total Energy Consumption: {tracker._total_energy.kWh}")
                        
                        accelerator.log(
                            {
                                "perplexity": perplexity,
                                "eval_loss": eval_loss,
                                "avg_train_loss": total_loss.item() / completed_steps,
                                "epoch": epoch + 1,
                                "step": completed_steps,
                            },
                            step=completed_steps,
                        )
                        
                        # Only the main process should log the validation metrics to wandb.
                        if accelerator.is_main_process:

                            if extra_args.wandb_token is not None:

                                    wandb.log({
                                        "eval_loss": eval_loss,
                                        "perplexity": perplexity,     
                                        "avg_train_loss": total_loss.item() / completed_steps,
                                        "total_energy_consumption": tracker._total_energy.kWh,      
                                    })
                
            # If we have reached the `max_steps`, break the step loop.
            if training_args.max_steps > 0 and completed_steps >= training_args.max_steps:
                break

        # If we have reached the `max_steps`, break the epoch loop
        if training_args.max_steps > 0 and completed_steps >= training_args.max_steps:
            break

    # Resume codecarbon tracking.
    tracker.stop()
    logger.info("Training complete!")

    # Upload the final emissions file to the Hub.
    if training_args.push_to_hub and training_args.hub_token is not None:
        if training_args.hub_model_id is not None:
            
            try:

                api.upload_file(
                    path_or_fileobj=f"{training_args.output_dir}/emissions_{accelerator.process_index}.csv",
                    path_in_repo=f"emissions_{accelerator.process_index}.csv",
                    repo_id=f"{training_args.hub_model_id}"
                )

                logger.info(f"Final emissions file pushed to the hub!")
            
            except Exception as e:
                logger.warning(f"Error while uploading emissions file to Hub: {e}")

    accelerator.wait_for_everyone()

    # Resume wandb tracking (only in the main process).
    if accelerator.is_main_process:
        if extra_args.wandb_token is not None:
            wandb.finish()
  
    # Save the final checkpoint at the end of training and push it to the Hub.
    if training_args.output_dir is not None:

        output_dir = os.path.join(training_args.output_dir, "final-checkpoint")
        accelerator.save_state(output_dir)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        tokenizer.save_pretrained(output_dir)

    if accelerator.is_main_process:

        if training_args.push_to_hub and training_args.hub_token is not None:
            if training_args.hub_model_id is not None:

                # Here we are going to push the model checkpoint to the HuggingFace Hub in a try-except block. 
                # If the push to the Hub fails, we will log a warning.
                try:
                    
                    # Push the final checkpoint to the Hub.
                    api.upload_folder(
                        repo_id=f"{training_args.hub_model_id}",  
                        folder_path=output_dir,
                    )

                    logger.info(f"Final model pushed to the hub!")
                                
                except Exception as e:
                    logger.warning(f"Error while uploading checkpoint to Hub: {e}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Llama 2 on a Brazilian Portuguese dataset.")
    parser.add_argument("--spec-file", help="Path to the spec YAML file")
    args = parser.parse_args()

    # Run the main function.
    main(args.spec_file)