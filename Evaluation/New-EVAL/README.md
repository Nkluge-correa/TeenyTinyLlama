# New Evaluations

We performed the following evaluations using a [Portuguese implementation of the EleutherAI LM Evaluation Harness](https://github.com/eduagarcia/lm-evaluation-harness-pt):

- [ENEM](https://ieeexplore.ieee.org/document/8247091) (3-shot) - The Exame Nacional do Ensino Médio (ENEM) is an advanced High-School level exam widely applied every year by the Brazilian government to students that wish to undertake a University degree. This dataset contains 1,430 questions that don't require image understanding of the exams from 2010 to 2018, 2022, and 2023. - Data sources:  [[1]](https://huggingface.co/datasets/eduagarcia/enem_challenge),  [[2]](https://www.ime.usp.br/~ddm/project/enem/),  [[3]](https://github.com/piresramon/gpt-4-enem),  [[4]](https://huggingface.co/datasets/maritaca-ai/enem). Metric - Accuracy.

- [BLUEX](https://arxiv.org/abs/2307.05410) (3-shot) - BLUEX is a multimodal dataset consisting of the two leading university entrance exams conducted in Brazil: Convest (Unicamp) and Fuvest (USP), spanning from 2018 to 2024. The benchmark comprises 724 questions that do not have accompanying images - Data sources:  [[1]](https://huggingface.co/datasets/eduagarcia-temp/BLUEX_without_images),  [[2]](https://github.com/portuguese-benchmark-datasets/bluex),  [[3]](https://huggingface.co/datasets/portuguese-benchmark-datasets/BLUEX). Metric - Accuracy.

- [OAB Exams](https://arxiv.org/abs/1712.05128) (3-shot) - OAB Exams is a dataset of more than 2,000 questions from the Brazilian Bar Association's exams from 2010 to 2018. - Data sources:  [[1]](https://huggingface.co/datasets/eduagarcia/oab_exams),  [[2]](https://github.com/legal-nlp/oab-exams). Metric - Accuracy.

- [ASSIN2 RTE](https://dl.acm.org/doi/abs/10.1007/978-3-030-41505-1_39) (15-shot) - ASSIN 2 (Avaliação de Similaridade Semântica e Inferência Textual - Evaluating Semantic Similarity and Textual Entailment) is the second edition of ASSIN, an evaluation shared task in the scope of the computational processing of Portuguese. Recognising Textual Entailment (RTE), also called Natural Language Inference (NLI), is the task of predicting if a given text (premise) entails (implies) another text (hypothesis). - Data sources:  [[1]](https://huggingface.co/datasets/eduagarcia/portuguese_benchmark),  [[2]](https://sites.google.com/view/assin2/),  [[3]](https://huggingface.co/datasets/assin2). Metric - F1-macro.

- [ASSIN2 STS](https://dl.acm.org/doi/abs/10.1007/978-3-030-41505-1_39) (15-shot) - Same as a dataset as above. Semantic Textual Similarity (STS) 'measures the degree of semantic equivalence between two sentences'. - Data sources:  [[1]](https://huggingface.co/datasets/eduagarcia/portuguese_benchmark),  [[2]](https://sites.google.com/view/assin2/),  [[3]](https://huggingface.co/datasets/assin2). Metric - Pearson.

- [FAQUAD NLI](https://ieeexplore.ieee.org/abstract/document/8923668) (15-shot) - FaQuAD is a Portuguese reading comprehension dataset that follows the format of the Stanford Question Answering Dataset (SQuAD). The dataset aims to address the problem of the abundance of questions sent by academics whose answers can be found in the available institutional documents in the Brazilian higher education system. It consists of 900 questions about 249 reading passages taken from 18 official documents of a computer science college from a Brazilian federal university and 21 Wikipedia articles related to the Brazilian higher education system. FaQuAD-NLI is a modified version of the FaQuAD dataset that repurposes the question-answering task as a textual entailment task between a question and its possible answers. - Data sources:  [[1]](https://github.com/liafacom/faquad/),  [[2]](https://huggingface.co/datasets/ruanchaves/faquad-nli). Metric - F1-macro.

- [HateBR](https://arxiv.org/abs/2103.14972) (25-shot) - HateBR is the first large-scale expert annotated dataset of Brazilian Instagram comments for abusive language detection on the web and social media. The HateBR was collected from politicians' Brazilian Instagram comments and manually annotated by specialists. It comprises 7,000 documents annotated with a binary classification (offensive versus non-offensive comments). - Data sources:  [[1]](https://huggingface.co/datasets/eduagarcia/portuguese_benchmark),  [[2]](https://github.com/franciellevargas/HateBR),  [[3]](https://huggingface.co/datasets/ruanchaves/hatebr). Metric - F1-macro.

The notebook used to run these evaluations is the [`lm-evaluation-harness-pt-br.ipynb`](./lm-evaluation-harness-pt-br.ipynb). Available on Colab. Full results are stored in the [results folder](./results/).

<a href="https://colab.research.google.com/drive/1m6Oqey4P9ShYTO62yRq7wrM_eEsvFJ9D" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>

## Benchmarks

|                    | **ASSIN2 RTE** | **ASSIN2 STS** | **BLUEX** | **ENEM** | **FAQUAD NLI** | **HateBR** | **OAB Exams** | **Average** |
|--------------------|----------------|----------------|-----------|----------|----------------|------------|---------------|-------------|
| **Qwen-1.8B**      | 64.83          | 19.53          | 26.15     | 30.23    | 43.97          | 33.33      | 27.20         | 35.03       |
| **TinyLlama-1.1B** | 58.93          | 13.57          | 22.81     | 22.25    | 43.97          | 36.92      | 23.64         | 31.72       |
| **TTL-460m**       | 53.93          | 12.66          | 22.81     | 19.87    | 49.01          | 33.59      | 27.06         | 31.27       |
| **XGLM-564m**      | 49.61          | 22.91          | 19.61     | 19.38    | 43.97          | 33.99      | 23.42         | 30.41       |
| **Bloom-1b7**      | 53.60          | 4.81           | 21.42     | 18.96    | 43.97          | 34.89      | 23.05         | 28.67       |
| **TTL-160m**       | 53.36          | 2.58           | 21.84     | 18.75    | 43.97          | 36.88      | 22.60         | 28.56       |
| **OPT-125m**       | 39.77          | 2.00           | 21.84     | 17.42    | 43.97          | 47.04      | 22.78         | 27.83       |
| **Pythia-160**     | 33.33          | 12.81          | 16.13     | 16.66    | 50.36          | 41.09      | 22.82         | 27.60       |
| **OLMo-1b**        | 34.12          | 9.28           | 18.92     | 20.29    | 43.97          | 41.33      | 22.96         | 27.26       |
| **TTL-460m-Chat**  | 43.39          | 4.84           | 23.23     | 19.38    | 33.98          | 33.49      | 26.97         | 26.46       |
| **Bloom-560m**     | 33.33          | 8.48           | 18.92     | 19.03    | 43.97          | 37.07      | 23.05         | 26.26       |
| **Pythia-410m**    | 33.33          | 4.80           | 19.47     | 19.45    | 43.97          | 33.33      | 23.01         | 25.33       |
| **OPT-350m**       | 33.33          | 3.65           | 20.72     | 17.35    | 44.71          | 33.33      | 23.01         | 25.15       |
| **GPT-2 small**    | 33.26          | 0.00           | 10.43     | 11.20    | 43.52          | 33.68      | 13.12         | 20.74       |
| **GPorTuguese**    | 33.33          | 3.85           | 14.74     | 3.01     | 28.81          | 33.33      | 21.23         | 19.75       |
| **Samba-1.1B**     | 33.33          | 1.30           | 8.07      | 10.22    | 17.72          | 35.79      | 15.03         | 17.35       |
