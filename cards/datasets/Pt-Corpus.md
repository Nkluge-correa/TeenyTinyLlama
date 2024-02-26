---
dataset_info:
  features:
  - name: text
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 16220765175.988096
    num_examples: 5768246
  download_size: 11478008666
  dataset_size: 16220765175.988096
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: other
task_categories:
- text-generation
language:
- pt
tags:
- portuguese
- language-modeling
pretty_name: Pt-Corpus
size_categories:
- 1M<n<10M
---

# Portuguese-Corpus

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://nkluge-correa.github.io/TeenyTinyLlama/
- **Repository:** https://github.com/Nkluge-correa/TeenyTinyLlama
- **Paper:** [TeenyTinyLlama: open-source tiny language models trained in Brazilian Portuguese](https://arxiv.org/abs/2401.16640)
- **Point of Contact:** [AIRES at PUCRS](mailto:nicholas@airespucrs.org)

### Dataset Summary

Portuguese-Corpus is a concatenation of several portions of Brazilian Portuguese datasets found in the [Hub](https://huggingface.co/datasets?task_categories=task_categories:text-generation&language=language:pt&sort=trending).

In a tokenized format, the dataset (uncompressed) weighs 50 GB and has approximately 4.1B tokens. This version does not have instructional content.

### Supported Tasks and Leaderboards

This dataset can be utilized for tasks involving language modeling.

### Languages

Portuguese.

## Dataset Structure

### Data Instances

The dataset consists of the following features:

- **text:** a string of text in Portuguese.
- **metadata:** the source where that string originated.

### Data Fields

```python
{
  "text": "A inteligência artificial (de sigla: IA; do inglês: artificial intelligence, de sigla: AI) é um campo de estudo multidisciplinar que abrange varias áreas do conhecimento.",
  "metadata": "source: https://huggingface.co/datasets/graelo/wikipedia"
}
```

### Data Splits

Available splits are `train`.

```python
from datasets import load_dataset

dataset = load_dataset("nicholasKluge/Pt-Corpus", split='train')

# If you don't want to download the entire dataset, set streaming to `True`
dataset = load_dataset("nicholasKluge/Pt-Corpus", split='train', streaming=True)

```

## Dataset Creation

### Curation Rationale

This dataset was developed as part of the [TeenyTinyLlama: open-source tiny language models trained in Brazilian Portuguese](https://arxiv.org/abs/2401.16640) paper. In this study, we document the development of open-foundation models tailored for use in low-resource settings, their limitations, and their benefits.

### Source Data

#### Initial Data Collection and Normalization

We utilized some of the filters used in Rae et al. ([2021](https://arxiv.org/abs/2112.11446)), besides using a [fine-tuned BERTimbau](https://huggingface.co/nicholasKluge/ToxicityModelPT) to exclude samples classified above a pre-defined toxicity threshold.

#### Who are the source language producers?

All text samples are native to Portuguese or translated from other languages to Portuguese (slight contamination of other languages should also be expected).

### Annotations

#### Annotation process

Portuguese-Corpus is a concatenation of several portions of Brazilian Portuguese datasets found in the [Hub](https://huggingface.co/datasets?task_categories=task_categories:text-generation&language=language:pt&sort=trending). We utilized some of the filters used in Rae et al. ([2021](https://arxiv.org/abs/2112.11446)), besides using a [fine-tuned BERTimbau](https://huggingface.co/nicholasKluge/ToxicityModelPT) to exclude samples classified above a pre-defined toxicity threshold.

#### Who are the annotators?

[Nicholas Kluge Corrêa](mailto:nicholas@airespucrs.org).

### Personal and Sensitive Information

This dataset, sourced from web scraping, may potentially contain personal and sensitive information, alongside offensive, toxic, and disturbing language.

## Considerations for Using the Data

### Social Impact of Dataset

The presence of personal and sensitive information within the dataset raises concerns about privacy and data protection, potentially leading to breaches of individuals' confidentiality and security. Furthermore, the inclusion of offensive, toxic, and disturbing language in the dataset poses risks of perpetuating harmful behaviors and attitudes, contributing to the normalization of hate speech and online toxicity. Therefore, careful handling and ethical considerations are essential to mitigate these potential social impacts and promote responsible dataset use.

### Discussion of Biases

The inclusion of offensive, toxic, and disturbing language in the dataset poses risks of perpetuating harmful behaviors and attitudes, contributing to the normalization of hate speech and online toxicity.

### Other Known Limitations

A significant portion of the data within the dataset has been translated using translation engines, potentially resulting in corrupted samples of both language and code. While useful for quickly converting text between languages, translation engines often struggle with accurately preserving the syntax, semantics, and context of programming languages. As a result, the translated code may contain errors, syntax inconsistencies, or even introduce vulnerabilities, rendering it unreliable or unusable for its intended purpose.

## Additional Information

### Dataset Curators

[Nicholas Kluge Corrêa](mailto:nicholas@airespucrs.org).

### Licensing Information

The following datasets (_only training splits are a part of the corpus_) and respective licenses form the Portuguese-Corpus:

- [Wikipedia](https://huggingface.co/datasets/graelo/wikipedia) (License: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/))

- [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) (License: [ODC-By](https://opendatacommons.org/licenses/by/1-0/), [cc0-1.0](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301#licensing-information))

- [OSCAR](https://huggingface.co/datasets/eduagarcia/OSCAR-2301-pt_dedup) (License: [cc0-1.0](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301#licensing-information))

- [CCc100](https://huggingface.co/datasets/eduagarcia/cc100-pt) (License: [Common Crawl terms of use](https://commoncrawl.org/terms-of-use/))

- [Roots Wikiquote](https://huggingface.co/datasets/bigscience-data/roots_pt_wikiquote) (License: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/))

- [Roots Ted Talks](https://huggingface.co/datasets/bigscience-data/roots_pt_ted_talks_iwslt) (License: [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en))

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
