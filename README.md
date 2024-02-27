<div align="center">

# TeenyTinyLlama: open-source _tiny_ language models trained in Brazilian Portuguese

[Hugging Face](https://huggingface.co/collections/nicholasKluge/teenytinyllama-6582ea8129e72d1ea4d384f1) | [Preprint](https://arxiv.org/abs/2401.16640) | [Demo](https://huggingface.co/spaces/nicholasKluge/TeenyTinyLlama-Chat)

</div>
<p align="center">
        <img src="./img/combined-logo.png" alt="An illustration of two adorable alpacas, one brown and the other orange, standing on a large red and white mushroom. The brown alpaca is wearing a monocle and the orange one is sporting a small hat. The mushroom is surrounded by grass and smaller mushrooms at the base." height="400">
</p>

Large language models (LLMs) have significantly advanced natural language processing, but their progress has yet to be equal across languages. While most LLMs are trained in high-resource languages like English, multilingual models generally underperform monolingual ones. Additionally, aspects of their multilingual foundation sometimes restrict the byproducts they produce, like computational demands and licensing regimes. In this study, we document the development of open-foundation models tailored for use in low-resource settings, their limitations, and their benefits. This is the _TeenyTinyLlama_ pair: two compact models for Brazilian Portuguese text generation. We release them under the permissive Apache 2.0 license on [GitHub](https://github.com/Nkluge-correa/TeenyTinyLlama) and [Hugging Face](https://huggingface.co/collections/nicholasKluge/teenytinyllama-6582ea8129e72d1ea4d384f1) for community use and further development.

## Intended Uses

The primary intended use of TeenyTinyLlama is to research the challenges related to developing language models for low-resource languages. Checkpoints saved during training are intended to provide a controlled setting for performing scientific experiments. You may also further fine-tune and adapt TeenyTinyLlama for deployment, as long as your use is following the Apache 2.0 license. If you decide to use pre-trained TeenyTinyLlama as a basis for your fine-tuned model, please conduct your own risk and bias assessment.

## Out-of-scope Use

TeenyTinyLlama is not intended for deployment. It is not a product and should not be used for human-facing interactions.

TeenyTinyLlama models are Brazilian Portuguese language only and are not suitable for translation or generating text in other languages.

TeenyTinyLlama has not been fine-tuned for downstream contexts in which language models are commonly deployed.

## Limitations

Like almost all other language models trained on large text datasets scraped from the web, the TTL pair exhibited behavior that does not make them an out-of-the-box solution to many real-world applications, especially those requiring factual, reliable, nontoxic text generation. Our models are all subject to the following:

- **Hallucinations:** This model can produce content that can be mistaken for truth but is, in fact, misleading or entirely false, i.e., hallucination.

- **Biases and Toxicity:** This model inherits the social and historical stereotypes from the data used to train it. Given these biases, the model can produce toxic content, i.e., harmful, offensive, or detrimental to individuals, groups, or communities.

- **Unreliable Code:** The model may produce incorrect code snippets and statements. These code generations should not be treated as suggestions or accurate solutions.

- **Language Limitations:** The model is primarily designed to understand standard Brazilian Portuguese. Other languages might challenge its comprehension, leading to potential misinterpretations or errors in response.

- **Repetition and Verbosity:** The model may get stuck on repetition loops (especially if the repetition penalty during generations is set to a meager value) or produce verbose responses unrelated to the prompt it was given.

Hence, even though our models are released with a permissive license, we urge users to perform their risk analysis on these models if intending to use them for real-world applications and also have humans moderating the outputs of these models in applications where they will interact with an audience, guaranteeing users are always aware they are interacting with a language model.

## Reproducing

This repository contains the source code used to train our models. We created all of our code implementations using the libraries tied to the Hugging Face ecosystem, i.e., [Transformers](https://github.com/huggingface/transformers), [Datasets](https://github.com/huggingface/datasets), [Tokenizers](https://github.com/huggingface/tokenizers), and [Accelerate](https://github.com/huggingface/accelerate), which allow for easy reproducibility, adaptation, and further scaling. Our training and evaluation scripts follow a standard [PyTorch](https://github.com/pytorch/pytorch) structure, while we utilized [CodeCarbon](https://github.com/mlco2/codecarbon) and [Weights & Biases](https://github.com/wandb/wandb) for tracking our experiments.

All requirements are listed in the requirements.txt file (Python version: 3.10.12).

- **Pre-training:** The Pre-training folder contains two main scripts: [`pre-training.py`](Pre-training/pre-training.py) and [`train-sentencepiece.py`](Pre-training/train-sentencepiece.py). These scripts were used to train both the Sentencepience tokenizer and the models. You can find more information on how to run them [here](Pre-training/README.md).

- **Fine-tuning:** The Fine-tuning folder contains the [`supervised-fine-tuning.py`](Fine-tuning/supervised-fine-tuning.py) script. This script is used to fine-tune the 460m version of our models on the [Instruct-Aira Dataset version 2.0](https://huggingface.co/datasets/nicholasKluge/instruct-aira-dataset-v2). You can find more information on how to run them [here](Fine-tuning/README.md).

- **Evaluation:** The Evaluation folder contains the results of our evaluations ([EVAL.md](Evaluation/EVAL.md)). It also contains an `evaluation.py` script to allow you to evaluate any of the checkpoints of our models or models you might come to train. The `lm-evaluation-harness-pt.ipynb` notebook showcases how to evaluate a model on the [Laiviet](https://github.com/laiviet/lm-evaluation-harness) version of the [`LM-Evaluation-Harness`](https://github.com/EleutherAI/lm-evaluation-harness). You can find more information on how to run them [here](Evaluation/README.md). Evaluations on Portuguese benchmarks are available in the [New-EVAL](Evaluation/New-EVAL) folder.

- **Utilities:** The Utilities folder contains some auxiliary scripts (more information available [here](Utilities/README.md)):
  
  - `chinchilla-estimation.py` helps you estimate dataset size concerning model size, using the [Chinchilla paper](https://arxiv.org/abs/2203.15556) as a reference.
  - `quantize.py` will perform 4-bit AWQ quantization on the models.
  - `tokenize-dataset.py` will create a tokenized version of a text dataset and upload it to the Hugging Face Hub.

In the `img` folder, you will find a subfolder named `logs and plots`. In it, you can find all the logs and plots (and the script used to make the plots) we used in our preprint.

## Cite as 🤗

```latex

@misc{correa24ttllama,
  title = {TeenyTinyLlama: open-source tiny language models trained in Brazilian Portuguese},
  author = {Corr{\^e}a, Nicholas Kluge and Falk, Sophia and Fatimah, Shiza and Sen, Aniket and De Oliveira, Nythamar},
  journal={arXiv preprint arXiv:2401.16640},
  year={2024}
}

```

## Funding

This research was funded by RAIES ([Rede de Inteligência Artificial Ética e Segura](https://www.raies.org/)). RAIES is a project supported by FAPERGS ([Fundação de Amparo à Pesquisa do Estado do Rio Grande do Sul](https://fapergs.rs.gov.br/inicial)) and CNPq ([Conselho Nacional de Desenvolvimento Científico e Tecnológico](https://www.gov.br/cnpq/)).

## License

TeenyTinyLlama is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more details.
