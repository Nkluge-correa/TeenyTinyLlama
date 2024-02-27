---
license: apache-2.0
datasets:
- nicholasKluge/Pt-Corpus-Instruct
language:
- pt
metrics:
- perplexity
library_name: transformers
pipeline_tag: text-generation
tags:
- text-generation-inference
widget:
- text: "A PUCRS é uma universidade "
  example_title: Exemplo
- text: "A muitos anos atrás, em uma galáxia muito distante, vivia uma raça de"
  example_title: Exemplo
- text: "Em meio a um escândalo, a frente parlamentar pediu ao Senador Silva para"
  example_title: Exemplo
inference:
  parameters:
    repetition_penalty: 1.2
    temperature: 0.2
    top_k: 20
    top_p: 0.2
    max_new_tokens: 150
co2_eq_emissions:
  emissions: 5.6
  source: CodeCarbon
  training_type: pre-training
  geographical_location: Germany
  hardware_used: NVIDIA A100-SXM4-40GB
---
# TeenyTinyLlama-160m

<img src="../../img/160m-llama.png" alt="A little llama wearing a mushroom hat and a monocle." height="200">

## Model Summary

Large language models (LLMs) have significantly advanced natural language processing, but their progress has yet to be equal across languages. While most LLMs are trained in high-resource languages like English, multilingual models generally underperform monolingual ones. Additionally, aspects of their multilingual foundation sometimes restrict the byproducts they produce, like computational demands and licensing regimes. Hence, we developed the _TeenyTinyLlama_ pair: two compact models for Brazilian Portuguese text generation.

Read our preprint on [ArXiv](https://arxiv.org/abs/2401.16640).

## Details

- **Architecture:** a Transformer-based model pre-trained via causal language modeling
- **Size:** 162,417,408 parameters
- **Context length:** 2048 tokens
- **Dataset:** [Pt-Corpus Instruct](https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-Instruct) (6.2B tokens)
- **Language:** Portuguese
- **Number of steps:** 458,000
- **GPU:** 1 NVIDIA A100-SXM4-40GB
- **Training time**: ~ 36 hours
- **Emissions:** 5.6 KgCO2 (Germany)
- **Total energy consumption:** 15.5 kWh

This repository has the [source code](https://github.com/Nkluge-correa/TeenyTinyLlama) used to train this model. The main libraries used are:

- [Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://github.com/pytorch/pytorch)
- [Datasets](https://github.com/huggingface/datasets)
- [Tokenizers](https://github.com/huggingface/tokenizers)
- [Sentencepiece](https://github.com/google/sentencepiece)
- [Accelerate](https://github.com/huggingface/accelerate)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [Codecarbon](https://github.com/mlco2/codecarbon)

## Intended Uses

The primary intended use of TeenyTinyLlama is to research the challenges related to developing language models for low-resource languages. Checkpoints saved during training are intended to provide a controlled setting for performing scientific experiments. You may also further fine-tune and adapt TeenyTinyLlama for deployment, as long as your use is following the Apache 2.0 license. If you decide to use pre-trained TeenyTinyLlama as a basis for your fine-tuned model, please conduct your own risk and bias assessment.

## Out-of-scope Use

TeenyTinyLlama is not intended for deployment. It is not a product and should not be used for human-facing interactions.

TeenyTinyLlama models are Brazilian Portuguese language only and are not suitable for translation or generating text in other languages.

TeenyTinyLlama has not been fine-tuned for downstream contexts in which language models are commonly deployed.

## Basic usage

Using the `pipeline`:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="nicholasKluge/TeenyTinyLlama-160m")

completions  = generator("Astronomia é a ciência", num_return_sequences=2, max_new_tokens=100)

for comp in completions:
  print(f"🤖 {comp['generated_text']}")
```

Using the `AutoTokenizer` and `AutoModelForCausalLM`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and the tokenizer
tokenizer = AutoTokenizer.from_pretrained("nicholasKluge/TeenyTinyLlama-160m", revision='main')
model = AutoModelForCausalLM.from_pretrained("nicholasKluge/TeenyTinyLlama-160m", revision='main')

# Pass the model to your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.eval()
model.to(device)

# Tokenize the inputs and pass them to the device
inputs = tokenizer("Astronomia é a ciência", return_tensors="pt").to(device)

# Generate some text
completions = model.generate(**inputs, num_return_sequences=2, max_new_tokens=100)

# Print the generated text
for i, completion in enumerate(completions):
    print(f'🤖 {tokenizer.decode(completion)}')
```

## Limitations

Like almost all other language models trained on large text datasets scraped from the web, the TTL pair exhibited behavior that does not make them an out-of-the-box solution to many real-world applications, especially those requiring factual, reliable, nontoxic text generation. Our models are all subject to the following:

- **Hallucinations:** This model can produce content that can be mistaken for truth but is, in fact, misleading or entirely false, i.e., hallucination.

- **Biases and Toxicity:** This model inherits the social and historical stereotypes from the data used to train it. Given these biases, the model can produce toxic content, i.e., harmful, offensive, or detrimental to individuals, groups, or communities.

- **Unreliable Code:** The model may produce incorrect code snippets and statements. These code generations should not be treated as suggestions or accurate solutions.

- **Language Limitations:** The model is primarily designed to understand standard Brazilian Portuguese. Other languages might challenge its comprehension, leading to potential misinterpretations or errors in response.

- **Repetition and Verbosity:** The model may get stuck on repetition loops (especially if the repetition penalty during generations is set to a meager value) or produce verbose responses unrelated to the prompt it was given.

Hence, even though our models are released with a permissive license, we urge users to perform their risk analysis on these models if intending to use them for real-world applications and also have humans moderating the outputs of these models in applications where they will interact with an audience, guaranteeing users are always aware they are interacting with a language model.

## Evaluations

During our training runs, both models showed consistent convergence. At no point did our evaluation curves show signs of overfitting or saturation. In the case of our 460m parameter model, we intentionally trained past the optimal point by approximately 75,000 steps to assess if there were any signs of saturation, but our evaluations consistently gave better results. We hypothesize that our models are under-trained but can improve if further trained to pass the Chinchilla optimal range.

| Processed Tokens | Perplexity | Energy Consumption (kWh)  | Emissions (KgCO2eq)  |
|------------------|------------|---------------------------|----------------------|
| 8.1M             | 20.49      | 9.40                      | 3.34                 |
| 1.6B             | 16.90      | 18.82                     | 6.70                 |
| 2.4B             | 15.43      | 28.59                     | 10.16                |
| 3.2B             | 14.64      | 38.20                     | 13.57                |
| 4.0B             | 14.08      | 48.04                     | 17.07                |
| 4.9B             | 13.61      | 57.74                     | 20.52                |
| 5.7B             | 13.25      | 67.32                     | 23.92                |
| 6.5B             | 12.87      | 76.84                     | 27.30                |
| 7.3B             | 12.57      | 86.40                     | 30.70                |
| 8.1B             | 12.27      | 96.19                     | 34.18                |
| 9.0B             | 11.96      | 106.06                    | 37.70                |
| 9.8B             | 11.77      | 115.69                    | 41.31                |

## Benchmarks

Evaluations on English benchmarks were performed using the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) (by [EleutherAI](https://www.eleuther.ai/)). [Laiviet](https://github.com/laiviet/lm-evaluation-harness) translated the tasks from the LM-Evaluation-Harness we used. The results of models marked with an "*" were extracted from the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

|                  | **ARC**   | **HellaSwag** | **MMLU**  | **TruthfulQA** | **Average** |
|------------------|-----------|---------------|-----------|----------------|-------------|
| Pythia-410m      | 24.83*    | 41.29*        | 25.99*    | 40.95*         | 33.26       |
| **TTL-460m**     | 29.40     | 33.00         | 28.55     | 41.10          | 33.01       |
| Bloom-560m       | 24.74*    | 37.15*        | 24.22*    | 42.44*         | 32.13       |
| Xglm-564M        | 25.56     | 34.64*        | 25.18*    | 42.53          | 31.97       |
| OPT-350m         | 23.55*    | 36.73*        | 26.02*    | 40.83*         | 31.78       |
| **TTL-160m**     | 26.15     | 29.29         | 28.11     | 41.12          | 31.16       |
| Pythia-160m      | 24.06*    | 31.39*        | 24.86*    | 44.34*         | 31.16       |
| OPT-125m         | 22.87*    | 31.47*        | 26.02*    | 42.87*         | 30.80       |
| GPorTuguese-2    | 22.48     | 29.62         | 27.36     | 41.44          | 30.22       |
| Gpt2-small       | 21.48*    | 31.60*        | 25.79*    | 40.65*         | 29.97       |
| Multilingual GPT | 23.81     | 26.37*        | 25.17*    | 39.62          | 28.73       |

Evaluations on Brazilian Portuguese benchmarks were performed using a [Portuguese implementation of the EleutherAI LM Evaluation Harness](https://github.com/eduagarcia/lm-evaluation-harness-pt) (created by [Eduardo Garcia](https://github.com/eduagarcia/lm-evaluation-harness-pt)).

|                | **ASSIN2 RTE** | **ASSIN2 STS** | **BLUEX** | **ENEM** | **FAQUAD NLI** | **HateBR** | **OAB Exams** | **Average** |
|----------------|----------------|----------------|-----------|----------|----------------|------------|---------------|-------------|
| Qwen-1.8B      | 64.83          | 19.53          | 26.15     | 30.23    | 43.97          | 33.33      | 27.20         | 35.03       |
| TinyLlama-1.1B | 58.93          | 13.57          | 22.81     | 22.25    | 43.97          | 36.92      | 23.64         | 31.72       |
| **TTL-460m**   | 53.93          | 12.66          | 22.81     | 19.87    | 49.01          | 33.59      | 27.06         | 31.27       |
| XGLM-564m      | 49.61          | 22.91          | 19.61     | 19.38    | 43.97          | 33.99      | 23.42         | 30.41       |
| Bloom-1b7      | 53.60          | 4.81           | 21.42     | 18.96    | 43.97          | 34.89      | 23.05         | 28.67       |
| **TTL-160m**   | 53.36          | 2.58           | 21.84     | 18.75    | 43.97          | 36.88      | 22.60         | 28.56       |
| OPT-125m       | 39.77          | 2.00           | 21.84     | 17.42    | 43.97          | 47.04      | 22.78         | 27.83       |
| Pythia-160     | 33.33          | 12.81          | 16.13     | 16.66    | 50.36          | 41.09      | 22.82         | 27.60       |
| OLMo-1b        | 34.12          | 9.28           | 18.92     | 20.29    | 43.97          | 41.33      | 22.96         | 27.26       |
| Bloom-560m     | 33.33          | 8.48           | 18.92     | 19.03    | 43.97          | 37.07      | 23.05         | 26.26       |
| Pythia-410m    | 33.33          | 4.80           | 19.47     | 19.45    | 43.97          | 33.33      | 23.01         | 25.33       |
| OPT-350m       | 33.33          | 3.65           | 20.72     | 17.35    | 44.71          | 33.33      | 23.01         | 25.15       |
| GPT-2 small    | 33.26          | 0.00           | 10.43     | 11.20    | 43.52          | 33.68      | 13.12         | 20.74       |
| GPorTuguese    | 33.33          | 3.85           | 14.74     | 3.01     | 28.81          | 33.33      | 21.23         | 19.75       |
| Samba-1.1B     | 33.33          | 1.30           | 8.07      | 10.22    | 17.72          | 35.79      | 15.03         | 17.35       |

## Fine-Tuning Comparisons

To further evaluate the downstream capabilities of our models, we decided to employ a basic fine-tuning procedure for our TTL pair on a subset of tasks from the Poeta benchmark. We apply the same procedure for comparison purposes on both [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased) models, given that they are also LLM trained from scratch in Brazilian Portuguese and have a similar size range to our models. We used these comparisons to assess if our pre-training runs produced LLM capable of producing good results ("good" here means "close to BERTimbau") when utilized for downstream applications.

| Models          | IMDB      | FaQuAD-NLI | HateBr    | Assin2    | AgNews    | Average |
|-----------------|-----------|------------|-----------|-----------|-----------|---------|
| BERTimbau-large | **93.58** | 92.26      | 91.57     | **88.97** | 94.11     | 92.10   |
| BERTimbau-small | 92.22     | **93.07**  | 91.28     | 87.45     | 94.19     | 91.64   |
| **TTL-460m**    | 91.64     | 91.18      | **92.28** | 86.43     | **94.42** | 91.19   |
| **TTL-160m**    | 91.14     | 90.00      | 90.71     | 85.78     | 94.05     | 90.34   |

All the shown results are the higher accuracy scores achieved on the respective task test sets after fine-tuning the models on the training sets. All fine-tuning runs used the same hyperparameters, and the code implementation can be found in the [model cards](https://huggingface.co/nicholasKluge/TeenyTinyLlama-460m-HateBR) of our fine-tuned models.

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

This repository was built as part of the RAIES ([Rede de Inteligência Artificial Ética e Segura](https://www.raies.org/)) initiative, a project supported by FAPERGS - ([Fundação de Amparo à Pesquisa do Estado do Rio Grande do Sul](https://fapergs.rs.gov.br/inicial)), Brazil.

## License

TeenyTinyLlama-160m is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more details.
