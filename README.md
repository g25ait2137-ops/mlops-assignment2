# MLOps Assignment 2 — Goodreads Genre Classification with DistilBERT

**Author:** Aishwarya Mishra (G25AIT2137) · MLOps · PGD AI Program · IIT Jodhpur

Fine-tunes `distilbert-base-cased` on the UCSD Goodreads reviews dataset to classify each review into one of 8 book genres. Training runs on a Kaggle GPU notebook, experiments are tracked with Weights & Biases, and the trained model + tokenizer are published to the Hugging Face Hub.

## Genres (labels)

`poetry`, `children`, `comics_graphic`, `fantasy_paranormal`, `history_biography`, `mystery_thriller_crime`, `romance`, `young_adult`

## Setup

```bash
git clone https://github.com/g25ait2137-ops/mlops-assignment2.git
cd mlops-assignment2
pip install -r requirements.txt
```

Set credentials (locally) or use Kaggle Secrets when running on Kaggle:

```bash
export WANDB_API_KEY=...   # from https://wandb.ai/settings
export HF_TOKEN=...        # from https://huggingface.co/settings/tokens
```

## Training platform: Kaggle Notebook

The model is trained inside a Kaggle Notebook using free GPU access (T4 x2). Reproducing the run:

1. Open Kaggle → New Notebook → File → Import Notebook → upload `kaggle_notebook.ipynb`.
2. Settings → **Accelerator: GPU T4 x2** · **Internet: ON**.
3. Add-ons → Secrets → add `WANDB_API_KEY` and `HF_TOKEN`, attach to notebook.
4. Run all cells.

Kaggle notebook (public): **[https://www.kaggle.com/code/aishwaryamishra8/kaggle-notebook-ipynb](https://www.kaggle.com/code/aishwaryamishra8/kaggle-notebook-ipynb)**

## Results


| Metric              | Score  |
| ------------------- | ------ |
| Accuracy            | 0.6006 |
| F1 Score (weighted) | 0.5972 |
| Eval Loss           | 2.2605 |


### Per-class F1 (test set, 200 reviews per class)


| Genre                  | Precision | Recall | F1   |
| ---------------------- | --------- | ------ | ---- |
| comics_graphic         | 0.81      | 0.79   | 0.80 |
| poetry                 | 0.74      | 0.82   | 0.78 |
| children               | 0.65      | 0.67   | 0.66 |
| history_biography      | 0.61      | 0.58   | 0.60 |
| romance                | 0.59      | 0.57   | 0.58 |
| mystery_thriller_crime | 0.51      | 0.61   | 0.56 |
| fantasy_paranormal     | 0.47      | 0.47   | 0.47 |
| young_adult            | 0.38      | 0.31   | 0.34 |


Full classification report: `[eval_report.json](eval_report.json)` (logged as a W&B Artifact).

## Links

- Hugging Face model: [https://huggingface.co/g25ait2137/distilbert-goodreads-genres](https://huggingface.co/g25ait2137/distilbert-goodreads-genres)
- W&B dashboard: [https://wandb.ai/g25ait2137-prom-iit-rajasthan/mlops-assignment2](https://wandb.ai/g25ait2137-prom-iit-rajasthan/mlops-assignment2)
- Kaggle notebook: **[https://www.kaggle.com/code/aishwaryamishra8/kaggle-notebook-ipynb](https://www.kaggle.com/code/aishwaryamishra8/kaggle-notebook-ipynb)**

## Repo structure

```
.
├── kaggle_notebook.ipynb   # Notebook to upload into Kaggle
├── kaggle_notebook.py      # Same code as a flat Python script (reference)
├── requirements.txt
├── eval_report.json        # Saved after training (from W&B artifact)
└── README.md
```

