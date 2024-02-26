# Running the Pre-training scripts

Documentation on how to run the pre-training scripts.

## Training a Sentencepiece tokenizer

Before running the `pre-training.py` file, you need a tokenizer. The [`train-sentencepiece.py`](train-sentencepiece.py) file allows you to train a [`PreTrainedTokenizerFast`](https://huggingface.co/docs/transformers/v4.37.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) using the original Llama 2 tokenizer configurations as a basis. You can run this script like this:

```bash
python train-sentencepiece.py \
--dataset_name "nicholasKluge/Pt-Corpus-Instruct" \
--dataset_split "train" \
--hub_token "hf_..." \
--reference_tokenizer "meta-llama/Llama-2-7b-hf" \
--num_samples 2000000 \
--vocab_size 32000
```

These are the arguments you pass to this script:

| Argument              | Description                                                                                                                                                                          |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dataset_name`        | The name of a Hugging Face dataset to use as training data. If you want to make a small test, we recommend [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k). |
| `dataset_split`       | The split of the dataset to use.                                                                                                                                                     |
| `hub_token`           | Your API key for the HUB (important if you want to use models or datasets that require authenticated access).                                                                        |
| `reference_tokenizer` | The tokenizer we will use to clone the configurations from.                                                                                                                          |
| `num_samples`         | The number of samples to use during training. If you try to run on more than 2M samples, you might run into OOM errors.                                                              |
| `vocab_size`          | Size of the vocabulary of the tokenizer you wish to train.                                                                                                                           |

> **Note: The script adds the following special tokens to the tokenizer: `["<unk>", "<s>", "</s>",  "<pad>"]`. If you don't wish to add them, you must modify the train-sentencepiece script.**

## Training a TeenyTinyLlama 🦙

Before you start training the models, you need to pre-process your dataset (tokenize and concatenate all strings into chunks of 2048 tokens). The datasets we used are available in text form and tokenized form in the URLs below:

- [Pt-Corpus](https://huggingface.co/datasets/nicholasKluge/Pt-Corpus) (raw text).
- [Pt-Corpus-Instruct](https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-Instruct) (raw text).
- [Pt-Corpus-tokenized](https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-tokenized) (tokenized - 4.1B tokens).
- [Pt-Corpus-Instruct-tokenized-small](https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-Instruct-tokenized-small) (tokenized - 3.7B tokens).
- [Pt-Corpus-Instruct-tokenized-large](https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-Instruct-tokenized-large) (tokenized - 6.2B tokens).

To speed up experiments, the `pre-training.py` script expects you to set a folder path where the dataset will be stored locally. The dataset folder must contain a list of parquet files, and you can achieve this by simply cloning the dataset from the hub to a local directory:

```bash
git lfs install
git clone https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-tokenized
```

Then, you should separate the dataset into `train` and `test` folders. Or you can modify the script and load the dataset like this. If the dataset is set to be saved in your cache folder, you will only need to download it once:

```python
train_dataset = load_dataset("nicholasKluge/Pt-Corpus-tokenized", split='train')
eval_dataset = load_dataset("nicholasKluge/Pt-Corpus-tokenized", split='test')
```

> **Note: Our scripts do not support [streaming](https://huggingface.co/docs/datasets/stream) since much of the arithmetic behind the stipulation of the training uses the length of the dataloaders as a factor. If you want to allow streaming (recommended for larger datasets, but it results in a slower training when compared to having the dataset loaded in memory), you will need to modify how these calculations are made by, for example, hard coding the number of steps, examples in each training split, etc.**

You are ready to start training if you have set up the dataset loading correctly. All configurations to train the model must be passed as a value in the [specs.yaml](specs.yaml) file. The [`specifications.py`](specifications.py) then correctly parses all the arguments used by `pre-training.py`.

These are the arguments you can modify in the specification file:

| Section            | Argument                      | Description                                                                                                                                                                                     |
|--------------------|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Model Arguments    | `model_to_train`              | The pre-trained model to use for training.                                                                                                                                                      |
|                    | `tokenizer_name`              | The name of the tokenizer associated with the pre-trained model.                                                                                                                                |
|                    | `model_id`                    | Identifier for the model. We use this to differentiate runs in the W&B dashboard.                                                                                                               |
|                    | `train_from_scratch`          | Whether to train the model from scratch or extend the training of an already pre-trained model.                                                                                                 |
|                    | `vocab_size`                  | Size of the vocabulary for the tokenizer.                                                                                                                                                       |
|                    | `hidden_size`                 | The dimensionality of the model, which is the dimensionality of the embedding layer.                                                                                                            |
|                    | `intermediate_size`           | Size of the intermediate layers in the model.                                                                                                                                                   |
|                    | `max_position_embeddings`     | Maximum number of positions for positional embeddings. This is the size of the context window of your transformer model.                                                                        |
|                    | `num_attention_heads`         | Number of attention heads in the model.                                                                                                                                                         |
|                    | `num_hidden_layers`           | Number of hidden layers in the model.                                                                                                                                                           |
|                    | `num_key_value_heads`         | Number of key and value heads in the model (used for grouped query attention).                                                                                                                  |
|                    | `output_hidden_states`        | Whether to output hidden states during training.                                                                                                                                                |
|                    | `cache_dir`                   | Directory to cache downloaded models and configurations.                                                                                                                                        |
|                    | `use_fast_tokenizer`          | Whether to use the fast tokenizer implementation.                                                                                                                                               |
|                    | `model_revision`              | Revision of the model to use.                                                                                                                                                                   |
|                    | `trust_remote_code`           | Whether to trust remote code when loading models.                                                                                                                                               |
|                    | `torch_dtype`                 | Data type for PyTorch tensors ('auto' for automatic selection).                                                                                                                                 |
|                    | `low_cpu_mem_usage`           | Option for low CPU memory usage.                                                                                                                                                                |
|                    | `use_cache`                   | Whether to use caching during training.                                                                                                                                                         |
|                    | `attn_implementation`         | Attention implementation for the model. FlashAttention 2 requires an Ampere GPU.                                                                                                                |
| Data Arguments     | `dataset_name`                | Name of the dataset to use for training.                                                                                                                                                        |
|                    | `folder_path`                 | Path to the folder containing the already tokenized dataset. The pre-training script does not perform tokenization! Use the [`tokenize-dataset.py`](../Utilities/tokenize-dataset.py) for this. |
|                    | `dataset_split`               | Split of the dataset to use (e.g., 'train').                                                                                                                                                    |
|                    | `block_size`                  | Maximum sequence length for training. We recommend to set this value as the same value for `max_position_embeddings`.                                                                           |
|                    | `overwrite_cache`             | Whether to overwrite the cached dataset.                                                                                                                                                        |
|                    | `preprocessing_num_workers`   | Number of workers for data preprocessing.                                                                                                                                                       |
|                    | `sanity_check`                | Whether to perform a sanity check. This runs the whole script using only a small portion of the dataset.                                                                                        |
| Training Arguments | `output_dir`                  | Directory to save training checkpoints.                                                                                                                                                         |
|                    | `do_train`                    | Whether to perform training.                                                                                                                                                                    |
|                    | `do_eval`                     | Whether to perform evaluation.                                                                                                                                                                  |
|                    | `evaluation_strategy`         | Strategy for evaluation options are 'steps' or 'no' for this script.                                                                                                                            |
|                    | `eval_steps`                  | Number of steps between evaluations.                                                                                                                                                            |
|                    | `per_device_train_batch_size` | Batch size per GPU for training.                                                                                                                                                                |
|                    | `per_device_eval_batch_size`  | Batch size per GPU for evaluation.                                                                                                                                                              |
|                    | `gradient_accumulation_steps` | Number of steps for gradient accumulation.                                                                                                                                                      |
|                    | `learning_rate`               | Initial learning rate for training.                                                                                                                                                             |
|                    | `adam_epsilon`                | Epsilon value for Adam optimizer.                                                                                                                                                               |
|                    | `weight_decay`                | Weight decay for optimizer.                                                                                                                                                                     |
|                    | `lr_scheduler_type`           | Type of learning rate scheduler (e.g., 'cosine').                                                                                                                                               |
|                    | `warmup_steps`                | Number of warmup steps for the learning rate scheduler.                                                                                                                                         |
|                    | `max_steps`                   | Maximum number of training steps.                                                                                                                                                               |
|                    | `num_train_epochs`            | Number of training epochs.                                                                                                                                                                      |
|                    | `gradient_checkpointing`      | Whether to use gradient checkpointing for memory efficiency.                                                                                                                                    |
|                    | `seed`                        | Random seed for reproducibility.                                                                                                                                                                |
|                    | `dataloader_pin_memory`       | Whether to pin memory in data loaders.                                                                                                                                                          |
|                    | `push_to_hub`                 | Whether to push the model checkpoints to the Hugging Face Hub.                                                                                                                                  |
|                    | `resume_from_checkpoint`      | Path to a checkpoint file in case you are resuming training. The folder must contain all states used by the Accelerator.                                                                        |
|                    | `hub_model_id`                | Identifier for the model on the Hugging Face Hub.                                                                                                                                               |
|                    | `hub_token`                   | API key for your Hugging Face account Hub.                                                                                                                                                         |
| Extra Arguments    | `wandb_token`                 | API key for Weights & Biases (WandB) integration.                                                                                                                                               |
|                    | `logger_name`                 | Name for the logger.                                                                                                                                                                            |
|                    | `wandb_log_steps`             | Number of steps between logging to WandB.                                                                                                                                                       |
|                    | `mixed_precision`             | Mixed precision training setting ('no' for no mixed precision).                                                                                                                                 |
|                    | `checkpointing_steps`         | Number of steps between saving checkpoints.                                                                                                                                                     |
|                    | `sample_every`                | Number of steps between generating text samples during training.                                                                                                                                |
|                    | `generation_seeds`            | List of text seeds for generating samples during training.                                                                                                                                      |

Having your specification file configured, you can run the pre-training script like this:

```bash
python pre-training.py --spec-file specs.yaml
```

For distributed training, run this script using [`Accelerate`](https://huggingface.co/docs/accelerate/index):

```bash
accelerate launch --num_processes=4 pre-training.py --spec-file specs.yaml
```

This will launch 4 processes on the current node, each with 1 GPU device per process. More information can be found [here](https://huggingface.co/docs/accelerate/basic_tutorials/launch).

> **Note: Although not required, we recommend running `accelerate config` before running `accelerate launch` to specify the configurations of your distributed environment. On the contrary, Accelerate will set default configurations based on your environment (e.g., the number of visible Cuda devises). These are saved in a `default_config.yaml` file and later used by Accelerate. You can get more information on the Accelerate [documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch#why-you-should-always-use-accelerate-config) and this [step-by-step tutorial](https://huggingface.co/blog/ram-efficient-pytorch-fsdp) from Hugging Face.**
