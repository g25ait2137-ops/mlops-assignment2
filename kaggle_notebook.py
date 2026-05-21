# =============================================================================
# MLOps Assignment 2 - Fine-Tuning DistilBERT for Goodreads Genre Classification
# Author: Aishwarya Mishra (G25AIT2137)
# =============================================================================


# ===== Cell 1: Install dependencies =====
get_ipython().system('pip install -q -U transformers wandb huggingface_hub gdown')


# ===== Cell 2: Load secrets from Kaggle =====
from kaggle_secrets import UserSecretsClient
import os

secrets = UserSecretsClient()
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
HF_TOKEN = os.environ["HF_TOKEN"]
print("Secrets loaded.")


# ===== Cell 3: Imports =====
import random
import json
import gzip
import pickle
import requests
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

import wandb
from huggingface_hub import login as hf_login

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ===== Cell 4: Config =====
MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 3e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
SAMPLE_PER_GENRE = 1000           # 800 train + 200 test per genre
WANDB_PROJECT = "mlops-assignment2"
WANDB_RUN_NAME = "distilbert-run-1"
HF_USERNAME = "g25ait2137"
HF_REPO_NAME = "distilbert-goodreads-genres"
HF_MODEL_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"

device_name = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device_name)


# ===== Cell 5: Load Goodreads reviews by genre =====
genre_url_dict = {
    "poetry":                 "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children":               "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic":         "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal":     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography":      "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance":                "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult":            "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}


def load_reviews(url, head=10000, sample_size=SAMPLE_PER_GENRE):
    reviews = []
    count = 0
    response = requests.get(url, stream=True)
    with gzip.open(response.raw, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            reviews.append(d["review_text"])
            count += 1
            if head is not None and count >= head:
                break
    return random.sample(reviews, min(sample_size, len(reviews)))


genre_reviews_dict = {}
for genre, url in genre_url_dict.items():
    print(f"Loading reviews for: {genre}")
    genre_reviews_dict[genre] = load_reviews(url, head=10000, sample_size=SAMPLE_PER_GENRE)

# Persist locally in case download fails on a re-run
pickle.dump(genre_reviews_dict, open("genre_reviews_dict.pickle", "wb"))


# ===== Cell 6: Train/test split + label maps =====
train_texts, train_labels = [], []
test_texts, test_labels = [], []

for genre, reviews in genre_reviews_dict.items():
    reviews = random.sample(reviews, min(SAMPLE_PER_GENRE, len(reviews)))
    split = int(0.8 * len(reviews))
    for r in reviews[:split]:
        train_texts.append(r)
        train_labels.append(genre)
    for r in reviews[split:]:
        test_texts.append(r)
        test_labels.append(genre)

unique_labels = sorted(set(train_labels))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

print(f"Train size: {len(train_texts)}, Test size: {len(test_texts)}")
print(f"Num labels: {len(unique_labels)} → {unique_labels}")


# ===== Cell 7: Tokenizer + encoding =====
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

train_labels_encoded = [label2id[y] for y in train_labels]
test_labels_encoded = [label2id[y] for y in test_labels]


# ===== Cell 8: Custom torch dataset =====
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = MyDataset(train_encodings, train_labels_encoded)
test_dataset = MyDataset(test_encodings, test_labels_encoded)


# ===== Cell 9: Load pre-trained model =====
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
).to(device_name)


# ===== Cell 10: Initialise W&B =====
wandb.login(key=os.environ["WANDB_API_KEY"])
wandb.init(
    project=WANDB_PROJECT,
    name=WANDB_RUN_NAME,
    config={
        "model": MODEL_NAME,
        "epochs": NUM_EPOCHS,
        "batch_size": TRAIN_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "dataset": "UCSD Goodreads",
        "platform": "Kaggle",
        "num_labels": len(id2label),
    },
)


# ===== Cell 11: Metrics + Trainer =====
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    logging_dir="./logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="wandb",
    run_name=WANDB_RUN_NAME,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)


# ===== Cell 12: Train =====
trainer.train()


# ===== Cell 13: Evaluate + log final metrics + artifact =====
eval_results = trainer.evaluate()
print(eval_results)

wandb.log({
    "final/loss": eval_results["eval_loss"],
    "final/accuracy": eval_results["eval_accuracy"],
    "final/f1": eval_results["eval_f1"],
})

preds = trainer.predict(test_dataset).predictions.argmax(-1)
labels = [item["labels"].item() for item in test_dataset]

report_dict = classification_report(
    labels, preds,
    target_names=[id2label[i] for i in range(len(id2label))],
    output_dict=True,
)

with open("eval_report.json", "w") as f:
    json.dump(report_dict, f, indent=2)

# Also save a human-readable text version
report_text = classification_report(
    labels, preds,
    target_names=[id2label[i] for i in range(len(id2label))],
)
with open("eval_report.txt", "w") as f:
    f.write(report_text)
print(report_text)

artifact = wandb.Artifact("eval-report", type="evaluation")
artifact.add_file("eval_report.json")
artifact.add_file("eval_report.txt")
wandb.log_artifact(artifact)


# ===== Cell 14: Push model + tokenizer to Hugging Face Hub =====
hf_login(token=HF_TOKEN)

model.push_to_hub(HF_MODEL_ID)
tokenizer.push_to_hub(HF_MODEL_ID)

hf_url = f"https://huggingface.co/{HF_MODEL_ID}"
wandb.run.summary["huggingface_model"] = hf_url
print("Model pushed to:", hf_url)


# ===== Cell 15: Finish W&B =====
wandb.finish()
