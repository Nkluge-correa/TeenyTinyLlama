---
license: apache-2.0
datasets:
- ruanchaves/hatebr
language:
- pt
metrics:
- accuracy
library_name: transformers
pipeline_tag: text-classification
tags:
- hate-speech
widget:
- text: "Não concordo com a sua opinião."
  example_title: Exemplo
- text: "Pega a sua opinião e vai a merda com ela!"
  example_title: Exemplo
---
# TeenyTinyLlama-160m-HateBR

TeenyTinyLlama is a pair of small foundational models trained in Brazilian Portuguese.

This repository contains a version of [TeenyTinyLlama-160m](https://huggingface.co/nicholasKluge/TeenyTinyLlama-160m) (`TeenyTinyLlama-160m-HateBR`) fine-tuned on the [HateBR dataset](https://huggingface.co/datasets/ruanchaves/hatebr).

## Details

- **Number of Epochs:** 3
- **Batch size:** 16
- **Optimizer:** `torch.optim.AdamW` (learning_rate = 4e-5, epsilon = 1e-8)
- **GPU:** 1 NVIDIA A100-SXM4-40GB

## Usage

Using `transformers.pipeline`:

```python
from transformers import pipeline

text = "Pega a sua opinião e vai a merda com ela!"

classifier = pipeline("text-classification", model="nicholasKluge/TeenyTinyLlama-160m-HateBR")
classifier(text)

# >>> [{'label': 'TOXIC', 'score': 0.9998729228973389}]
```

## Reproducing

To reproduce the fine-tuning process, use the following code snippet:

```python

# Hatebr
! pip install transformers datasets evaluate accelerate -q

import evaluate
import numpy as np
from huggingface_hub import login
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load the task
dataset = load_dataset("ruanchaves/hatebr")

# Format the dataset
train = dataset['train'].to_pandas()
train = train[['instagram_comments', 'offensive_language']]
train.columns = ['text', 'labels']
train.labels = train.labels.astype(int)
train = Dataset.from_pandas(train)

test = dataset['test'].to_pandas()
test = test[['instagram_comments', 'offensive_language']]
test.columns = ['text', 'labels']
test.labels = test.labels.astype(int)
test = Dataset.from_pandas(test)

dataset = DatasetDict({
    "train": train,  
    "test": test                  
})

# Create a `ModelForSequenceClassification`
model = AutoModelForSequenceClassification.from_pretrained(
    "nicholasKluge/TeenyTinyLlama-160m", 
    num_labels=2, 
    id2label={0: "NONTOXIC", 1: "TOXIC"}, 
    label2id={"NONTOXIC": 0, "TOXIC": 1}
)

tokenizer = AutoTokenizer.from_pretrained("nicholasKluge/TeenyTinyLlama-160m")

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

dataset_tokenized = dataset.map(preprocess_function, batched=True)

# Create a simple data collactor
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Use accuracy as evaluation metric
accuracy = evaluate.load("accuracy")

# Function to compute accuracy
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="checkpoints",
    learning_rate=4e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
    hub_token="your_token_here",
    hub_model_id="username/model-ID",
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train!
trainer.train()

```

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

@misc{correa24ttllama,
  doi = {10.1016/j.mlwa.2024.100558},
  url = {https://www.sciencedirect.com/science/article/pii/S2666827024000343},
  title = {TeenyTinyLlama: open-source tiny language models trained in Brazilian Portuguese},
  author = {Corr{\^e}a, Nicholas Kluge and Falk, Sophia and Fatimah, Shiza and Sen, Aniket and De Oliveira, Nythamar},
  journal={Machine Learning With Applications},
  publisher = {Springer},
  year={2024}
}
```

## Funding

This repository was built as part of the RAIES ([Rede de Inteligência Artificial Ética e Segura](https://www.raies.org/)) initiative, a project supported by FAPERGS - ([Fundação de Amparo à Pesquisa do Estado do Rio Grande do Sul](https://fapergs.rs.gov.br/inicial)), Brazil.

## License

TeenyTinyLlama-160m-HateBR is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more details.
