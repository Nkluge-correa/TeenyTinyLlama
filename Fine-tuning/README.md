# Running the Fine-tuning script

The `supervised-fine-tuning.py` can be used to fine-tune the 460m version of our models on the [Instruct-Aira Dataset version 2.0](https://huggingface.co/datasets/nicholasKluge/instruct-aira-dataset-v2). You could also use the same script to fine-tune other models on other datasets. However, you might need to perform some minor modifications to the data preprocessing steps.

All configurations to fine-tune the model must be passed as a value in the [supervised-fine-tuning-specs.yaml](supervised-fine-tuning-specs.yaml) file. The [`specifications.py`](specifications.py) then correctly parses all the arguments used by `supervised-fine-tuning.py`.

These are the arguments you can modify in the specification file:

| Section            | Argument                      | Description                                                                                                             |
|--------------------|-------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| Model Arguments    | `base_model`                  | The pre-trained model to used in the fine-tuning                                                                        |
|                    | `model_id`                    | Identifier for the model. We use this to differentiate runs in the W&B dashboard                                        |
|                    | `use_fast`                    | Whether to use the fast tokenization implementation                                                                     |
|                    | `output_hidden_states`        | Whether to output hidden states during training                                                                         |
|                    | `cache_dir`                   | Directory to cache downloaded models and configurations                                                                 |
|                    | `model_revision`              | Revision of the model to use                                                                                            |
|                    | `trust_remote_code`           | Whether to trust remote code when loading models                                                                        |
|                    | `low_cpu_mem_usage`           | Option for low CPU memory usage                                                                                         |
|                    | `boi_token`                   | Beginning-of-instruction token                                                                                          |
|                    | `eoi_token`                   | End-of-instruction token                                                                                                |
|                    | `chat_template`               | Template for chat conversation during training                                                                          |
|                    | `attn_implementation`         | Attention implementation for the model                                                                                  |
| Data Arguments     | `dataset_name`                | Name of the dataset to use for training                                                                                 |
|                    | `dataset_split`               | Split of the dataset to use (e.g., 'train')                                                                             |
|                    | `block_size`                  | Maximum sequence length for training. We recommend setting this value as the same value for `max_position_embeddings`   |
|                    | `overwrite_cache`             | Whether to overwrite the cached dataset                                                                                 |
|                    | `preprocessing_num_workers`   | Number of workers for data preprocessing                                                                                |
|                    | `sanity_check`                | Whether to perform a sanity check                                                                                       |
| Training Arguments | `output_dir`                  | Directory to save training checkpoints                                                                                  |
|                    | `num_train_epochs`            | Number of training epochs                                                                                               |
|                    | `do_train`                    | Whether to perform training                                                                                             |
|                    | `do_eval`                     | Whether to perform evaluation                                                                                           |
|                    | `per_device_train_batch_size` | Batch size per GPU for training                                                                                         |
|                    | `per_device_eval_batch_size`  | Batch size per GPU for evaluation                                                                                       |
|                    | `gradient_accumulation_steps` | Number of steps for gradient accumulation                                                                               |
|                    | `learning_rate`               | Initial learning rate for training                                                                                      |
|                    | `adam_epsilon`                | Epsilon value for Adam optimizer                                                                                        |
|                    | `weight_decay`                | Weight decay for optimizer                                                                                              |
|                    | `lr_scheduler_type`           | Type of learning rate scheduler (e.g., 'cosine')                                                                        |
|                    | `warmup_steps`                | Number of warmup steps for the learning rate scheduler                                                                  |
|                    | `max_steps`                   | Maximum number of training steps                                                                                        |
|                    | `gradient_checkpointing`      | Whether to use gradient checkpointing for memory efficiency                                                             |
|                    | `seed`                        | Random seed for reproducibility                                                                                         |
|                    | `dataloader_pin_memory`       | Whether to pin memory in data loaders                                                                                   |
|                    | `push_to_hub`                 | Whether to push the model checkpoints to the Hugging Face Hub                                                           |
|                    | `resume_from_checkpoint`      | Path to a checkpoint file in case you are resuming training. The folder must contain all states used by the Accelerator |
|                    | `hub_model_id`                | Identifier for the model on the Hugging Face Hub                                                                        |
|                    | `hub_token`                   | API key for your Hugging Face account Hub                                                                               |
| Extra Arguments    | `wandb_token`                 | API key for Weights & Biases (WandB) integration                                                                        |
|                    | `logger_name`                 | Name for the logger                                                                                                     |
|                    | `wandb_log_steps`             | Number of steps between logging to WandB                                                                                |
|                    | `sample_every`                | Number of steps between generating text samples during training                                                         |
|                    | `mixed_precision`             | Mixed precision training setting ('no' for no mixed precision)                                                          |

Having your specification file configured, you can run the fine-tuning script like this:

```bash
python supervised-fine-tuning.py --spec-file supervised-fine-tuning-specs.yaml
```

For distributed training, run this script using `accelerate`:

```bash
accelerate launch --num_processes=4 supervised-fine-tuning.py --spec-file supervised-fine-tuning-specs.yaml
```

This will launch 4 processes on the current node, each with 1 GPU device per process. More information can be found [here](https://huggingface.co/docs/accelerate/basic_tutorials/launch).
