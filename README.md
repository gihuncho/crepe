<h1><img src="https://gihuncho.github.io/crepe/static/images/favicon.ico" alt="logo" style="height:30px"> CREPE</h1>

This repository contains the official Python package for **CREPE** — a fast, interpretable, and clinically grounded metric for automated chest X‑ray report evaluation.

<div style='display:flex; gap: 0.25rem; '>
<a href='https://gihuncho.github.io/crepe/'><img src='https://img.shields.io/badge/Project-URL-magenta'></a>
<a href='https://huggingface.co/gihuncho/crepe-biomedbert'><img src='https://img.shields.io/badge/Model-HuggingFace-yellow'></a>
<a href='https://gihuncho.github.io/crepe/static/pdfs/EMNLP2025_CREPE.pdf'><img src='https://img.shields.io/badge/Paper-PDF-blue'></a>
</div>

## Overview

![CREPE Overview](https://gihuncho.github.io/crepe/static/images/overview_new.png)
**What CREPE does.** Given a *reference* report and a *candidate* (generated) report, CREPE predicts **continuous error counts** over six clinically meaningful categories (A–F) and returns their **sum as the CREPE score** (lower is better). The model is a domain‑specific BERT encoder with **six regression heads** (plus auxiliary presence heads used in training).



## Install

```bash
# Python >= 3.9
pip install --upgrade pip
pip install "torch>=2.1" "transformers>=4.41"
````

This repo is a small Python package; you can import it directly from the project root (editable install optional).

> The default checkpoint is hosted on the Hugging Face Hub (`jogihood/crepe-biomedbert`) and will be downloaded on first use. If your environment has restricted internet access, pass a local `cache_dir` or model path (see below).



## Quickstart

```python
import torch
import crepe

# (optional) set a cache directory for HF model files
cache_dir = "your/path/to/cache/dir"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Load model & tokenizer (defaults to jogihood/crepe-biomedbert)
model, tokenizer = crepe.load_model_and_tokenizer(cache_dir=cache_dir)
model.to(device).eval()

# 2) Score a pair of reports
reference = "Normal chest radiograph"
candidate = "Bilateral pleural effusions noted"

result = crepe.compute(
    model=model,
    tokenizer=tokenizer,
    reference_report=reference,
    candidate_report=candidate,
    device=device.type,  # "cuda" or "cpu"
)

print(result)
# {
#   "crepe_score": <float>,  # lower is better
#   "predicted_error_counts": [nA, nB, nC, nD, nE, nF]  # continuous, >= 0
# }
```

> Tokenization uses pair encoding: `(reference, candidate)`, with truncation/padding to 512 tokens to match the training setup. 



## Interpreting the Outputs

* `predicted_error_counts`: a list of **six non‑negative floats** `[nA, nB, nC, nD, nE, nF]`, ordered by categories **A→F** as defined above. Values are **continuous** (not forced to integers).
* `crepe_score`: the **unweighted sum** of the six predicted counts. **Lower is better** (fewer predicted discrepancies). 



## API

```python
# crepe.__init__.py
load_model_and_tokenizer(model_name_or_path="jogihood/crepe-biomedbert", cache_dir=None)
# -> (model, tokenizer)

compute(model, tokenizer, reference_report, candidate_report, device="cpu")
# -> {"crepe_score": float, "predicted_error_counts": List[float]}
```

Advanced utilities (optional):

```python
# crepe.models.get_model_and_tokenizer(ckpt_path, device=None)
# crepe.models.get_predicted_counts(model, tokenizer, gt, pred, device=None)
```

> Under the hood, the model is a `PreTrainedModel` with six regression heads; auxiliary presence heads exist for training but are **not used at inference**. Predictions are clipped to be non‑negative. See [`crepe/models.py`](./crepe/models.py).


## Contact

If you have any questions, feel free to contact!

gihuncho@snu.ac.kr


## Citing


```bibtex
TBA
```
