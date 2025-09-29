---
layout: project_page
permalink: /

title: CREPE - Rapid Chest X-ray Report Evaluation by Predicting Multi-category Error Counts
authors:
    Gihun Cho, Seunghyun Jang, Hanbin Ko, Inhyeok Baek, Chang Min Park
affiliations:
    Seoul National University
paper: https://github.com/gihuncho/crepe/blob/main/EMNLP2025_CREPE.pdf
code: https://github.com/gihuncho/crepe
model: https://huggingface.co/gihuncho/crepe-biomedbert
---

<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
We introduce **CREPE** (Rapid **C**hest X-ray **R**eport **E**valuation by **P**redicting Multi-category **E**rror Counts), a rapid, interpretable, and clinically grounded metric for automated chest X-ray report generation. CREPE uses a domain-specific BERT model fine-tuned with a multi-head regression architecture to predict error counts across six clinically meaningful categories. Trained on a large-scale synthetic dataset of 32,000 annotated report pairs, CREPE demonstrates strong generalization and interpretability. On the expert-annotated ReXVal dataset, CREPE achieves a Kendall's $\tau$ correlation of 0.786 with radiologist error counts, outperforming traditional and recent metrics. CREPE achieves these results with an inference speed approximately 280 times faster than large language model (LLM)-based approaches, enabling rapid and fine-grained evaluation for scalable development of chest X-ray report generation models.
        </div>
    </div>
</div>

---
![CREPE]({{ '/static/image/overview_new.png' | relative_url }})

## Background
TODO

## Objective
TODO

## Key Ideas
TODO

## Significance
TODO

## Citation
```
@article{turing1936computable,
  title={On computable numbers, with an application to the Entscheidungsproblem},
  author={Turing, Alan Mathison},
  journal={Journal of Mathematics},
  volume={58},
  number={345-363},
  pages={5},
  year={1936}
}
```

> Note: This is an example of a Jekyll-based project website template: [Github link](https://github.com/shunzh/project_website).