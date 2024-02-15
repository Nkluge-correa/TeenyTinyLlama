---
dataset_info:
  features:
  - name: input_ids
    sequence: int32
  - name: attention_mask
    sequence: int8
  - name: labels
    sequence: int64
  splits:
  - name: train
    num_bytes: 48793769228.0
    num_examples: 1831873
  - name: test
    num_bytes: 479448000.0
    num_examples: 18000
  download_size: 14600379883
  dataset_size: 49273217228.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test
license: other
task_categories:
- text-generation
language:
- pt
tags:
- portuguese
- language-modeling
pretty_name: Pt-Corpus Instruct tokenized small
size_categories:
- 1M<n<10M
---

# Portuguese-Corpus Instruct (tokenized small)

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://nkluge-correa.github.io/TeenyTinyLlama/
- **Repository:** https://github.com/Nkluge-correa/TeenyTinyLlama
- **Paper:** [TeenyTinyLlama: open-source tiny language models trained in Brazilian Portuguese](https://arxiv.org/abs/2401.16640)
- **Point of Contact:** [AIRES at PUCRS](mailto:nicholas@airespucrs.org)

### Dataset Summary

This repository has a tokenized version (using the [TeenyTinyLlama tokenizer](https://huggingface.co/nicholasKluge/TeenyTinyLlama-460m)) of a small subset (3.7B tokens) of the [Pt-Corpus Instruct dataset](https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-Instruct). All sequences are 2048 tokens long. All sequences are 2048 tokens long. This dataset was used in "_[TeenyTinyLlama: open-source tiny language models trained in Brazilian Portuguese](https://arxiv.org/abs/2401.16640)_".

For more information, see the [original dataset card](https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-Instruct).

## Languages

Portuguese.

## Dataset Structure

### Data Instances

The dataset consists of the following features:

- **input_ids:** sequence of tokens.
- **attention_mask:** binary tensor indicating the position of the padded indices.
- **labels:** sequence of tokens.

### Data Fields

```python
{
  "input_ids": [ 1026, 1531, 1009, 8067,...],
  "attention_mask": [1, 1, 1, 1, ...],
  "labels": [ 1026, 1531, 1009, 8067,...]
}  
```

### Data Splits

Available splits are `train` (~ 1.8M) and `test` (18K).

```python
from datasets import load_dataset

dataset = load_dataset("nicholasKluge/Pt-Corpus-Instruct-tokenized-small", split='train')

# If you don't want to download the entire dataset, set streaming to `True`
dataset = load_dataset("nicholasKluge/Pt-Corpus-Instruct-tokenized-small", split='train', streaming=True)

```

## Additional Information

### Dataset Curators

[Nicholas Kluge Corrêa](mailto:nicholas@airespucrs.org).

### Citation Information

```latex

@misc{correa24ttllama,
  title = {TeenyTinyLlama: open-source tiny language models trained in Brazilian Portuguese},
  author = {Corr{\^e}a, Nicholas Kluge and Falk, Sophia and Fatimah, Shiza and Sen, Aniket and De Oliveira, Nythamar},
  journal={arXiv preprint arXiv:2401.16640},
  year={2024}
}

```

### Contributions

If you would like to contribute, contact me at [nicholas@airespucrs.org](mailto:nicholas@airespucrs.org)!
