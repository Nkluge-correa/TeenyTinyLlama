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
# python supervised-fine-tuning.py --spec-file specs.yaml
#
# For distributed training, run this script using `accelerate`:
#
# accelerate launch --config_file "fsdp_config.yaml" --num_processes=4 supervised-fine-tuning.py --spec-file specs.yaml
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
import math
import yaml
import torch
import wandb
import random
import logging
import argparse
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import AddedToken
from codecarbon import EmissionsTracker
from torch.utils.data import DataLoader
from huggingface_hub import create_repo, HfApi

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
    get_scheduler,
    GenerationConfig,
)

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed

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

    # Create a HuggingFace repository if needed (only the main process should do this).
    if accelerator.is_main_process:
        if training_args.push_to_hub and training_args.hub_token is not None:
            if training_args.hub_model_id is not None:
                create_repo(
                    repo_id=training_args.hub_model_id, 
                    token=training_args.hub_token,
                    repo_type="model",
                    exist_ok=True,
                    private=True)
            
            else:
                raise ValueError("No model id provided. Try running with `hub_model_id=your-user-name/your-model-name`")   

    # Load the fine-tuning dataset.
    if data_args.dataset_name is not None:

        dataset = load_dataset(
            data_args.dataset_name, 
            split=data_args.dataset_split,
            token=training_args.hub_token if training_args.hub_token else None,
            cache_dir=model_args.cache_dir,
        )

        # Make a list of prompts to serve as seeds for generation.
        seeds = [model_args.boi_token + x[0]['content'] + model_args.eoi_token for x in dataset.select(range(100))['conversations']]

        # Shuffle the dataset.
        dataset = dataset.shuffle(seed=training_args.seed)        

        # Sanity check: use only the first 100 examples
        if data_args.sanity_check:
            dataset = dataset.select(range(100))

            logger.info(f"Sanity check: using only the first 100 examples")

        logger.info(f"Loaded dataset: {data_args.dataset_name} | Split: {data_args.dataset_split} | Number of examples: {len(dataset):,}")

    else:

        raise ValueError("No dataset provided. Try running with `dataset_name=nicholasKluge/instruct-aira-dataset`")

    if model_args.base_model is not None:

        # Now we are going to load the configuration, model and tokenizer from the HuggingFace Hub.
        # According to the documentation, the `from_pretrained` methods guarantee that only one local process can concurrently
        # download the model/tokenizer from the HuggingFace Hub.
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.base_model, 
            **{
                "cache_dir": model_args.cache_dir,
                "use_fast": model_args.use_fast,
                "revision": model_args.model_revision,
                "token": training_args.hub_token,
                "trust_remote_code": model_args.trust_remote_code,
            }
        )

        # Add special tokens.
        special_tokens_dict = {
            "additional_special_tokens": [
                AddedToken(model_args.boi_token, lstrip=False, rstrip=False, normalized=True, single_word=False),
                AddedToken(model_args.eoi_token, lstrip=False, rstrip=False, normalized=True, single_word=False),
            ]
        }

        tokenizer.add_special_tokens(special_tokens_dict)

        logger.info(f"Special tokens added to the tokenizer: {tokenizer.all_special_tokens}")
        
        # Add chat template to the tokenizer.
        tokenizer.chat_template = model_args.chat_template
     
        # Load the configuration of the `base_model`
        configuration = AutoConfig.from_pretrained(
            model_args.base_model, 
            **{
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "token": training_args.hub_token,
                "trust_remote_code": model_args.trust_remote_code,
                "output_hidden_states": model_args.output_hidden_states,
            }
        )

        # Load the pretrained model to be fine-tuned
        model = AutoModelForCausalLM.from_pretrained(
                model_args.base_model,
                config=configuration,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=training_args.hub_token,
                trust_remote_code=model_args.trust_remote_code,
                low_cpu_mem_usage=model_args.low_cpu_mem_usage,
                attn_implementation=model_args.attn_implementation,
            )
              
        # Resize the token embeddings of the model to match the tokenizer.
        model.resize_token_embeddings(len(tokenizer))

        # Add new `name_or_path` to the model config.
        if training_args.hub_model_id is not None:
            model.config.name_or_path = training_args.hub_model_id

        # Gradient checkpointing can be enabled to reduce the memory usage during training.
        # However, this will slow down the training process by about 20%.
        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        
        logger.info(f"Model to train (base architecture): {model_args.base_model}")

    else:

        raise ValueError("No base model provided. Try running with `base_model=gpt2`")

    
    # Create a formated Chat column.
    dataset = dataset.map(lambda x: {"formatted_conversations": tokenizer.apply_chat_template(x["conversations"], tokenize=False, add_generation_prompt=False)})
    column_names = dataset.column_names

    # Tokenize all texts in the dataset.
    def tokenize_function(examples):
        return tokenizer(examples['formatted_conversations'],
            add_special_tokens=False,
            truncation=True,
            max_length=data_args.max_length,
            padding="max_length",
            )

    with accelerator.main_process_first():
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on every text in dataset",
        )

    # Add a column named `labels` wich is a copy of the `input_ids` column.
    with accelerator.main_process_first():
        dataset = dataset.map(
            lambda examples: {"labels": examples["input_ids"]},
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=True,
            desc="Adding labels to the dataset",
        )

    # Split the dataset into train and validation sets.
    if training_args.do_eval and data_args.validation_split_percentage is not None:

        logger.info("Splitting the dataset into train and validation sets...")

        dataset = dataset.train_test_split(test_size=data_args.validation_split_percentage)

        logger.info(f"Train set size: {len(dataset['train']):,} | Validation set size: {len(dataset['test']):,}")

    else:

        logger.info(f"Using the whole dataset for training. Training set size: {len(dataset):,}")

    # Create the Training DataLoader.
    if training_args.do_train and training_args.do_eval:
        if "train" not in dataset:
            raise ValueError("`do_train=True` requires a train dataset")
        train_dataset = dataset["train"]
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True, 
            collate_fn=default_data_collator, 
            batch_size=training_args.per_device_train_batch_size,
            pin_memory=training_args.dataloader_pin_memory,
        )

        # Create the Evaluation DataLoader.
        if "test" not in dataset:
            raise ValueError("`do_eval=True` requires a validation dataset")
        eval_dataset = dataset["test"] 
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=default_data_collator, 
            batch_size=training_args.per_device_eval_batch_size,
            pin_memory=training_args.dataloader_pin_memory,
        )
    
    elif training_args.do_train and not training_args.do_eval:
        train_dataset = dataset
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

    # Set max_steps
    training_args.max_steps = int(num_update_steps_per_epoch * training_args.num_train_epochs)

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
                notes="Fine tuning TeenyTinyLlama",
                tags=["Alignment", "Fine-tuning", "Energy Consumption", "Language Modeling", "Portuguese"],
                config=all_kwargs,
                name=f"""{extra_args.logger_name.lower()}-{model_args.model_id}-Chat-{time.strftime("%d-%m-%Y")}""",
            )

    # We would also like to track the energy consumption of the training process. We are going to use the `codecarbon` library
    # to do this. We need to initialize the `EmissionsTracker` and then track the energy consumption of the training process.
    tracker = EmissionsTracker(
        project_name=extra_args.logger_name,
        log_level="critical", # set to "critical" to silence codecarbon
        output_dir=training_args.output_dir,
        output_file=f"emissions_{accelerator.process_index}.csv",
        tracking_mode='machine'
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
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(training_args.max_steps), disable=not accelerator.is_local_main_process, unit=" samples", desc="Training")
    completed_steps = 0
    starting_epoch = 0

    # Update the progress_bar if load from checkpoint.
    progress_bar.update(completed_steps)

    # Start training loop and activate codecarbon tracking.
    tracker.start()

    for epoch in range(starting_epoch, training_args.num_train_epochs):

        logger.info(f'Beginning epoch {epoch + 1} of {training_args.num_train_epochs}')
        
        # Set the model to training mode.
        model.train()
        total_loss = 0
    
        # Iterate over the batches of data in the current epoch.
        for step, batch in enumerate(train_dataloader, start=1):
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
            
            accelerator.wait_for_everyone()
            
            # Generate text from the model every `sample_every ` steps.
            if accelerator.is_main_process:

                if completed_steps % extra_args.sample_every == 0 and not step == 0:
                    
                    model.config.use_cache = True

                    try:

                        model.eval()

                        if accelerator.is_main_process:
                            
                            # Sample a string from the `seeds` and generate text from the model.
                            inputs = tokenizer(random.choice(seeds), return_tensors="pt").to('cuda:0')
                                
                            sample_outputs = model.generate(**inputs,
                                                do_sample=True,
                                                top_k=50,
                                                max_new_tokens=150,
                                                top_p=0.50,
                                                repetition_penalty=1.2,
                                                num_return_sequences=5)
                            
                            model.config.use_cache = False
                            
                            texts = []

                            for i, sample_output in enumerate(sample_outputs):
                                texts.append(tokenizer.decode(sample_output))
                            
                            # Log the samples to the main process terminal.
                            for text in texts:
                                logger.info(f"Samples (Epoch: {epoch + 1} | Step: {step}): {text}")
                            
                            # Log the samples to wandb.
                            if extra_args.wandb_token is not None:

                                training_samples = wandb.Table(columns=[f"Samples (Epoch: {epoch + 1} | Step: {step})"])
                                for text in texts:
                                    training_samples.add_data(text)
                                wandb.log({f"Samples (Epoch: {epoch + 1} | Step: {step})": training_samples})
                        
                    except Exception as e:
                        logger.warning(f"Error while generating samples: {e}")
                        model.config.use_cache = False

                    model.train()
            
        # Evaluate the model at the end of each epoch if `do_eval=True`
        if training_args.do_eval:
            model.eval()
            losses = []
            logger.info(f"Running evaluation at the end of Epoch {epoch + 1}.")
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
            
            logger.info(f"Epoch {epoch + 1} | Perplexity: {perplexity} | Average Training Loss: {total_loss.item() / completed_steps} | Evaluation Loss: {eval_loss} | Total Energy Consumption: {tracker._total_energy.kWh}")
            
            # Only the main process should log the validation metrics to wandb.
            if accelerator.is_main_process:
                
                if extra_args.wandb_token is not None:

                        wandb.log({
                            "eval_loss": eval_loss,
                            "perplexity": perplexity,     
                            "avg_train_loss": total_loss.item() / completed_steps,
                            "total_energy_consumption": tracker._total_energy.kWh,      
                        })
            
        else:
            logger.info(f"Epoch {epoch + 1} | Average Training Loss: {total_loss.item() / completed_steps} | Total Energy Consumption: {tracker._total_energy.kWh}")

            if accelerator.is_main_process:
            
                if extra_args.wandb_token is not None:

                        wandb.log({    
                            "avg_train_loss": total_loss.item() / completed_steps,
                            "total_energy_consumption": tracker._total_energy.kWh,      
                        })
        
        # Save the model checkpoint at the end of each epoch.    
        accelerator.wait_for_everyone()

        output_dir = f"epoch_{epoch + 1}"

        if training_args.output_dir is not None:
            # Join the output directory with the current checkpoint directory.
            output_dir = os.path.join(training_args.output_dir, output_dir)
        # Save the model checkpoint.
        accelerator.save_state(output_dir)

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            training_args.output_dir, 
            is_main_process=accelerator.is_main_process, 
            save_function=accelerator.save
        )
        tokenizer.save_pretrained(output_dir)

        # Save the `generation_config` file
        generation_config = GenerationConfig(
            bos_token_id=tokenizer.bos_token_id,
            sep_token_id=tokenizer.sep_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            unk_token_id=tokenizer.unk_token_id,
            max_new_tokens=model.config.max_position_embeddings,
            min_length=0,
            do_sample=True,
            use_cache=False,
            renormalize_logits=True,
            top_k=30,
            top_p=0.3,
            temperature=0.3,
            repetition_penalty=1.2,
        )

        generation_config.save_pretrained(output_dir)

        tracker.flush()
        
    # Resume codecarbon tracking.
    tracker.stop()
    logger.info("Training complete!")
    i# Resume wandb tracking (only in the main process).
    if accelerator.is_main_process:
        if extra_args.wandb_token is not None:
            wandb.finish()

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
    parser = argparse.ArgumentParser(description="Fine tune a language model on an instruction dataset.")
    parser.add_argument("--spec-file", help="Path to the spec YAML file")
    args = parser.parse_args()
    main(args.spec_file)
