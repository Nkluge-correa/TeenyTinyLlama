from typing import Optional
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_to_train: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model to use as a basis for the training. Or, if `train_from_scratch` is set to `True`, ",
                "the model to get the configuration from to train from scratch."
            )
        },
    )

    model_id: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model id of the model to train.",
                "Options are: `160m` and `460m`.",
                "Caution: different `model_id` require different model configurations.",
                "We use these ids to differentiate between runs on the WandB dashboard."
            )    
        },
    )

    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path."},
    )

    train_from_scratch: bool = field(
        default=False,
        metadata={"help": "Whether to train the model from scratch."},
    )

    vocab_size: Optional[int] = field(
        default=32000,
        metadata={"help": "The vocab size of the tokenizer."},
    )

    hidden_size: Optional[int] = field(
        default=768,
        metadata={
            "help": (
                "The hidden size of the model. Only used if `train_from_scratch` is set to `True`."
                "For `model_id=160m`, we used `hidden_size` of `768`."
                "For `model_id=460m`, we used `hidden_size` of `1024`."
            )
        },
    )

    intermediate_size: Optional[int] = field(
        default=3072,
        metadata={
            "help": (
                "The intermediate size of the model. Only used if `train_from_scratch` is set to `True`."
                "For `model_id=160m`, we used `intermediate_size` of `3072`."
                "For `model_id=460m`, we used `intermediate_size` of `4096`."
            )
        },
    )

    max_position_embeddings: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum sequence length that this model might ever be used with. Only used if `train_from_scratch` is set to `True`."},
    )

    num_attention_heads: Optional[int] = field(
        default=12,
        metadata={
            "help": (
                "The number of attention heads used by the model. Only used if `train_from_scratch` is set to `True`."
                "For `model_id=160m`, we used `num_attention_heads` of `12`."
                "For `model_id=460m`, we used `num_attention_heads` of `16`."
            )
        },
    )

    num_hidden_layers: Optional[int] = field(
        default=12,
        metadata={
            "help": (
                "The number of hidden layers used by the model. Only used if `train_from_scratch` is set to `True`."
                "For `model_id=160m`, we used `num_hidden_layers` of `12`."
                "For `model_id=460m`, we used `num_hidden_layers` of `24`."
            )    
        },
    )

    num_key_value_heads: Optional[int] = field(
        default=12,
        metadata={
            "help": (
                "The number of key-value attention heads used by the model. Only used if `train_from_scratch` is set to `True`."
                "For `model_id=160m`, we used `num_key_value_heads` of `12`."
                "For `model_id=460m`, we used `num_key_value_heads` of `16`."
            )
        },
    )

    output_hidden_states: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to return all hidden-states (i.e., all hidden-states for all layers)."},
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )

    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )

    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    use_cache: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether or not to cache the loaded pretrained weights. Set to `False` to avoid caching when loading a model."
            )
        },
    )

    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Whether to use the optimized implementation of attention. "
                "Option is `None` or `flash_attention_2`."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, 
        metadata={
            "help": (
                "The name of the dataset to use (via the datasets library)."
                "The `pre-training.py` script does not support the data preprocessing."
                "Hence, the dataset must be already tokenized."
                "For this, you can use the `tokenize_dataset.py` script."
            )
        }
    )

    folder_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The path to the folder containing the dataset."
                "The `pre-training.py` script does not support the data preprocessing."
                "Hence, the dataset must be already tokenized."
                "For this, you can use the `tokenize_dataset.py` script."
            )
        }
    )

    dataset_split: Optional[str] = field(
        default="train",
        metadata={"help": "The dataset split to use."},
    )

    block_size: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )

    overwrite_cache: Optional[bool] = field(
        default=False, 
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    sanity_check: Optional[bool] = field(
        default=False,
        metadata={"help": "If set, will run training on a small portion of the dataset."},
    )

@dataclass
class ExtraArguments:
    """
    Arguments pertaining miscellaneous things (e.g., the Accelerator, W&B, logger name, etc.).
    """
    wandb_token: Optional[str] = field(
        default=None,
        metadata={"help": "The token to use for logging to wandb."},
    )

    logger_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the logger to use."},
    )

    wandb_log_steps: Optional[int] = field(
        default=1,
        metadata={"help": "The number of steps to log to wandb."},
    )

    with_tracking: Optional[bool] = field(
        default=True,
        metadata={"help": "Let the Accelerator track the model and optimizer states."},
    )

    sample_every: Optional[int] = field(
        default=100,
        metadata={"help": "The number of steps between each time the model generates samples."},
    )

    mixed_precision: Optional[str] = field(
        default='no',
        metadata={"help": "Whether to use mixed precision or not ('no', 'fp16', `bf16`)."},
    )

    checkpointing_steps: Optional[int] = field(
        default=None,
        metadata={"help": "The number of steps the various states should be saved at the end of every n steps."},
    )

    generation_seeds: Optional[list] = field(
        default=None,
        metadata={"help": "The generation seeds to use."},
    )