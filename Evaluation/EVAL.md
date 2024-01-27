# Evaluations

During our training runs, both models showed consistent convergence. At no point did our evaluation curves show signs of overfitting or saturation. In the case of our 460m parameter model, we intentionally trained past the optimal point by approximately 75,000 steps to assess if there were any signs of saturation, but our evaluations consistently gave better results. We hypothesize that our models are under-trained but can improve if further trained to pass the Chinchilla optimal range.

**TeenyTinyLlama-160m:**

| Processed Tokens   | Perplexity | Total Energy Consumption (kWh) | Emissions (KgCO2eq) |
|--------------------|------------|--------------------------------|---------------------|
| 8.1M               | 24.52      | 3.75                           | 1.28                |
| 1.6B               | 20.58      | 7.51                           | 2.56                |
| 2.4B               | 16.98      | 11.25                          | 3.84                |
| 3.2B               | 16.41      | 14.52                          | 5.11                |

**TeenyTinyLlama-460m:**

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

Evaluations on benchmarks were performed using the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) (by [EleutherAI](https://www.eleuther.ai/)). [Laiviet](https://github.com/laiviet/lm-evaluation-harness) translated the tasks from the LM-Evaluation-Harness we used:

- [ARC-Challenge](https://arxiv.org/abs/1803.05457v1): a multiple-choice question-answering dataset containing questions from early grades science exams.
- [HellaSwag](https://arxiv.org/abs/1905.07830): a multiple choice dataset for evaluating grounded commonsense inference.
- [MMLU](https://arxiv.org/abs/2009.03300): a benchmark that covers 57 subjects across STEM, humanities, social sciences, and more, measuring the performance of models on various natural language tasks.
- [TruthfulQA](https://arxiv.org/abs/2109.07958): a benchmark comprised of several questions, spanning 38 topics, that assess the model's tendency to replicate commonly believed falsehoods.

The results of models marked with an "*" were extracted from the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

|                  | **ARC**   | **HellaSwag** | **MMLU**  | **TruthfulQA** | **Average** |
|------------------|-----------|---------------|-----------|----------------|-------------|
| Pythia-410m      | 24.83*    | **41.29***    | 25.99*    | 40.95*         | 33.26       |
| **TTL-460m**     | **29.40** | 33.00         | **28.55** | 41.10          | 33.01       |
| Bloom-560m       | 24.74*    | 37.15*        | 24.22*    | 42.44*         | 32.13       |
| Xglm-564M        | 25.56     | 34.64*        | 25.18*    | **42.53**      | 31.97       |
| OPT-350m         | 23.55*    | 36.73*        | 26.02*    | 40.83*         | 31.78       |
| **TTL-160m**     | 26.15     | 29.29         | 28.11     | 41.12          | 31.16       |
| Pythia-160m      | 24.06*    | 31.39*        | 24.86*    | 44.34*         | 31.16       |
| OPT-125m         | 22.87*    | 31.47*        | 26.02*    | 42.87*         | 30.80       |
| GPorTuguese-2    | 22.48     | 29.62         | 27.36     | 41.44          | 30.22       |
| Gpt2-small       | 21.48*    | 31.60*        | 25.79*    | 40.65*         | 29.97       |
| Multilingual GPT | 23.81     | 26.37*        | 25.17*    | 39.62          | 28.73       |

## Fine-Tuning Comparisons

To further evaluate the downstream capabilities of our models, we decided to employ a basic fine-tuning procedure for our TTL pair on a subset of tasks from the Poeta benchmark. We apply the same procedure for comparison purposes on both [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased) models, given that they are also LLM trained from scratch in Brazilian Portuguese and have a similar size range to our models. We used these comparisons to assess if our pre-training runs produced LLM capable of producing good results ("good" here means "close to BERTimbau") when utilized for downstream applications.

| Models          | IMDB      | FaQuAD-NLI | HateBr    | Assin2    | AgNews    | Average |
|-----------------|-----------|------------|-----------|-----------|-----------|---------|
| BERTimbau-large | **93.58** | 92.26      | 91.57     | **88.97** | 94.11     | 92.10   |
| BERTimbau-small | 92.22     | **93.07**  | 91.28     | 87.45     | 94.19     | 91.64   |
| **TTL-460m**    | 91.64     | 91.18      | **92.28** | 86.43     | **94.42** | 91.19   |
| **TTL-160m**    | 91.14     | 90.00      | 90.71     | 85.78     | 94.05     | 90.34   |

All the shown results are the higher accuracy scores achieved on the respective task test sets after fine-tuning the models on the training sets. All fine-tuning runs used the same hyperparameters, and the code implementation can be found in the [model cards](https://huggingface.co/nicholasKluge/TeenyTinyLlama-460m-HateBR) of our fine-tuned models.
