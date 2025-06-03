# Running the Evaluation scripts

The pre-training script will perform evaluations if you set the `do_eval` argument to `True` and `evaluation_strategy` to `step`. However, you can also re-run evaluations, or run them separately, by using the `evaluation.py` script.

To speed up experiments, the `evaluation.py` scripts expects you to set a folder path where the dataset will be stored locally. The dataset folder must contain a list of parquet files, and you can achieve this by simply cloning the dataset from the hub to a local directory:

```bash
git lfs install
git clone https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-tokenized
```

Then, you should separate the dataset into `train` and `test` folders. Or you can modify the script and load the dataset like this. If the dataset is set to be saved in your cache folder, you will only need to download it once:

```python
eval_dataset = load_dataset("nicholasKluge/Pt-Corpus-tokenized", split='test')
```

> **Note: Our scripts do not support [streaming](https://huggingface.co/docs/datasets/stream) since much of the arithmetic behind the stipulation of the training uses the length of the dataloaders as a factor. If you want to allow streaming (recommended for larger datasets, but it results in a slower training when compared to having the dataset loaded in memory), you will need to modify how these calculations are made by, for example, hard coding the number of steps, examples in each training split, etc.**

You can run this script like this:

```bash
python evaluation.py \
--logger_name "TeenyTinyLlama" \
--model_checkpoint_path "nicholasKluge/TeenyTinyLlama-460m" \
--revision "step100000" \
--attn_implementation "flash_attention_2" \
--per_device_eval_batch_size 16 \
--completed_steps 100000 \
--total_energy_consumption 3.34
```

These are the arguments you pass to this script:

| Argument                     | Description                                                    |
| ---------------------------- | -------------------------------------------------------------- |
| `logger_name`                | The logger name                                                |
| `model_checkpoint_path`      | Path to the model checkpoint to be used for evaluation         |
| `revision`                   | Specify the revision for the model (e.g., "step100000")        |
| `attn_implementation`        | Specify the attention implementation for evaluation            |
| `per_device_eval_batch_size` | Set the batch size per device for evaluation                   |
| `completed_steps`            | Specify the number of completed training steps (e.g., 100000). |
| `total_energy_consumption`   | Specify the total energy consumption made thus far             |

## Benchmark Evaluation

- Learn how to run the evaluations using the [`LM-Evaluation-Harness`](https://github.com/EleutherAI/lm-evaluation-harness) (English) here 👉 <a href="https://colab.research.google.com/drive/1FvcrJRyc1fv8jS-g-OkT_tsBrKe3DfCX" target="_blank"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"> </a>.
- Learn how to run the evaluations using a fork of the [`LM-Evaluation-Harness`](https://github.com/EleutherAI/lm-evaluation-harness) (Multilingual) here 👉 <a href="https://colab.research.google.com/drive/1mspcStRItqKzLZ39PG-ztKJXCqSvlEKt" target="_blank"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"> </a>.
- Learn how to run the evaluations using a fork of the [`LM-Evaluation-Harness`](https://github.com/EleutherAI/lm-evaluation-harness) (Portuguese) here 👉 <a href="https://colab.research.google.com/drive/1m6Oqey4P9ShYTO62yRq7wrM_eEsvFJ9D" target="_blank"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"> </a>.
- Learn how to run the Alpaca-Eval (Portuguese) here 👉 <a href="https://colab.research.google.com/drive/1jUszJNk7ik0CTUZD_dzxvnN_YsoxcPky" target="_blank"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"> </a>.
