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
    num_bytes: 80805366840.0
    num_examples: 3033690
  - name: test
    num_bytes: 799080000.0
    num_examples: 30000
  download_size: 24242436009
  dataset_size: 81604446840.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
license: other
task_categories:
- text-generation
language:
- pt
tags:
- portuguese
- language-modeling
pretty_name: Pt-Corpus Instruct tokenized large
size_categories:
- 1M<n<10M
---

# Portuguese-Corpus Instruct (tokenized large)

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

This repository has a tokenized version (using the [TeenyTinyLlama tokenizer](https://huggingface.co/nicholasKluge/TeenyTinyLlama-460m)) of the [Portuguese-Corpus Instruct dataset](https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-Instruct). All sequences are 2048 tokens long. All sequences are 2048 tokens long. This dataset was used in "_[TeenyTinyLlama: open-source tiny language models trained in Brazilian Portuguese](https://arxiv.org/abs/2401.16640)_".

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

Available splits are `train` (~ 3M) and `test` (30K).

```python
from datasets import load_dataset

dataset = load_dataset("nicholasKluge/Pt-Corpus-Instruct-tokenized-large", split='train')

# If you don't want to download the entire dataset, set streaming to `True`
dataset = load_dataset("nicholasKluge/Pt-Corpus-Instruct-tokenized-large", split='train', streaming=True)

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
