# Running the Utilities

Documentation on how to run the utility scripts.

## Chinchilla Estimation

The `chinchilla-estimation.py` will estimate your model's training using the [Chinchilla paper](https://arxiv.org/abs/2203.15556) as a reference. You must change the `N` and `D` parameters and run the script to see the estimations. The `calculate_loss` function estimates the foreseen loss for a language model of size `N` trained on a size `D` dataset.

## Quantize Models

The `quantize.py` will perform 4-bit AWQ quantization on the models. You can run this script like this:

```bash
python quantize.py \
--token "hf_..." \
--model_path "nicholasKluge/TeenyTinyLlama-460m" \
--quant_path "TeenyTinyLlama-460m-awq" \
```

These are the arguments you pass to this script:

| Argument     | Description                                            |
|--------------|--------------------------------------------------------|
| `token`      | API key for your Hugging Face account                  |
| `model_path` | Path to the model checkpoint to be used for evaluation |
| `quant_path` | Name of a local folder to save the quantize model      |

## Tokenize the Dataset

Before you start training the models, you need to pre-process your dataset (tokenize and concatenate all strings into chunks of 2048 tokens). Our tokenized datasets are available in the URLs below:

- [Pt-Corpus-tokenized](https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-tokenized) (tokenized - 4.1B tokens).
- [Pt-Corpus-Instruct-tokenized-small](https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-Instruct-tokenized-small) (tokenized - 3.7B tokens).
- [Pt-Corpus-Instruct-tokenized-large](https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-Instruct-tokenized-large) (tokenized - 6.2B tokens).

If you would like to create another tokenized dataset, you can use the `tokenize-dataset.py` script. It will create a tokenized version of a text dataset and upload it to the Hugging Face Hub. You can run this script like this:

```bash
python tokenize-dataset.py \
--dataset-name "nicholasKluge/Pt-Corpus" \
--dataset-split "train" \
--tokenizer-name "nicholasKluge/TeenyTinyLlama-460m" \
--block-size 2048 \
--test-size 30000 \
--shuffle True \
--seed 42 \
--token "hf_..."
```

These are the arguments you pass to this script:

| Argument         | Description                                        |
|------------------|----------------------------------------------------|
| `dataset-name`   | Name of the dataset to tokenize                    |
| `dataset-split`  | Split of the dataset to tokenize (e.g., 'train')   |
| `tokenizer-name` | Name of the tokenizer to use for tokenization      |
| `block-size`     | Maximum sequence length for tokenization           |
| `test-size`      | Size of the test set for tokenization              |
| `shuffle`        | Whether to shuffle the dataset during tokenization |
| `seed`           | Random seed for reproducibility                    |
| `token`          | Token for authentication on the Hugging Face Hub   |
