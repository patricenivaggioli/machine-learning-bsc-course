# Appendix E — What to Learn Next

> **You've completed the foundations!** Here's a roadmap for going further — from intermediate topics to advanced specialisations.

---

## E.1 Immediate Next Steps

These topics build directly on what you've learned:

| Topic | What it adds | Where to start |
|:------|:------------|:---------------|
| **Regularisation** (Ridge, Lasso, Elastic Net) | Prevents overfitting in linear models by penalising large weights | scikit-learn docs → Linear Models |
| **Support Vector Machines (SVM)** | Powerful classifier that finds the maximum-margin boundary | scikit-learn docs → SVM |
| **XGBoost / LightGBM** | Industry-standard gradient boosting libraries, faster and more flexible | XGBoost documentation |
| **Feature selection** | Choosing the most informative features systematically | scikit-learn → Feature selection |
| **Handling imbalanced data** | SMOTE, class weights, under/oversampling | imbalanced-learn library |

---

## E.2 Deep Learning Path

If you want to work with images, text, or audio:

```mermaid
flowchart TD
    A["This guide<br/>(ML foundations)"] --> B["Deep Learning basics<br/>(PyTorch or TensorFlow)"]
    B --> CV["Computer Vision<br/>(CNNs)"]
    B --> NLP["Natural Language<br/>Processing<br/>(Transformers)"]
    B --> GEN["Generative AI<br/>(GANs, Diffusion,<br/>Large Language Models)"]
    CV --> ADV1["Object detection<br/>Segmentation"]
    NLP --> ADV2["BERT, GPT<br/>Fine-tuning LLMs"]
    GEN --> ADV3["Stable Diffusion<br/>ChatGPT-style apps"]

    style A fill:#74b9ff,stroke:#0984e3,color:#000
    style B fill:#ffeaa7,stroke:#fdcb6e,color:#000
    style CV fill:#55efc4,stroke:#00b894,color:#000
    style NLP fill:#a29bfe,stroke:#6c5ce7,color:#000
    style GEN fill:#fab1a0,stroke:#e17055,color:#000
    style ADV1 fill:#dfe6e9,stroke:#636e72,color:#000
    style ADV2 fill:#dfe6e9,stroke:#636e72,color:#000
    style ADV3 fill:#dfe6e9,stroke:#636e72,color:#000
```

---

## E.3 MSc-Level Study Guide

For a more comprehensive and mathematically rigorous treatment of all topics in this guide — plus advanced chapters on SVMs, regularisation, Bayesian methods, time series, and reinforcement learning — see the companion guide:

> **Machine Learning — A Comprehensive Study Guide** (MSc level)
>
> Located in: `machine-learning-msc-course/`

---

## E.4 Recommended Books

| Book | Level | Focus |
|:-----|:------|:------|
| *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (Aurélien Géron) | Beginner–Intermediate | Practical ML and deep learning |
| *An Introduction to Statistical Learning* (James, Witten, Hastie, Tibshirani) | Beginner–Intermediate | Theory with R/Python labs (free PDF) |
| *Deep Learning with Python* (François Chollet) | Intermediate | Keras/TensorFlow deep learning |
| *The Hundred-Page Machine Learning Book* (Andriy Burkov) | Beginner | Concise overview |
| *Pattern Recognition and Machine Learning* (Christopher Bishop) | Advanced | Mathematical foundations |

---

## E.5 Online Courses

| Course | Platform | Notes |
|:-------|:---------|:------|
| **Machine Learning** (Andrew Ng) | Coursera | The classic intro course |
| **Deep Learning Specialisation** (Andrew Ng) | Coursera | 5-course deep learning series |
| **Fast.ai — Practical Deep Learning** | fast.ai | Top-down, code-first approach |
| **CS229: Machine Learning** | Stanford (YouTube) | University-level theory |
| **CS231n: CNNs for Visual Recognition** | Stanford (YouTube) | Computer vision |
| **CS224n: NLP with Deep Learning** | Stanford (YouTube) | Natural language processing |
| **Kaggle Learn** | kaggle.com | Short, hands-on micro-courses |

---

## E.6 Practice Platforms

| Platform | What you'll do |
|:---------|:--------------|
| **Kaggle** | Competitions, datasets, notebooks, community |
| **Google Colab** | Free GPU notebooks for experiments |
| **HuggingFace** | Pre-trained models, datasets, Transformers library |
| **UCI ML Repository** | Classic benchmark datasets |
| **Papers With Code** | Find state-of-the-art methods with implementations |

---

## E.7 Topics by Career Interest

```mermaid
flowchart LR
    subgraph DS["Data Scientist"]
        D1["Statistics & A/B testing"]
        D2["Feature engineering"]
        D3["Business communication"]
    end
    subgraph MLE["ML Engineer"]
        M1["Model deployment (MLOps)"]
        M2["Docker, APIs, cloud"]
        M3["Monitoring & retraining"]
    end
    subgraph RS["Research Scientist"]
        R1["Advanced maths"]
        R2["Read papers"]
        R3["Reproduce experiments"]
    end

    style D1 fill:#74b9ff,stroke:#0984e3,color:#000
    style D2 fill:#74b9ff,stroke:#0984e3,color:#000
    style D3 fill:#74b9ff,stroke:#0984e3,color:#000
    style M1 fill:#55efc4,stroke:#00b894,color:#000
    style M2 fill:#55efc4,stroke:#00b894,color:#000
    style M3 fill:#55efc4,stroke:#00b894,color:#000
    style R1 fill:#a29bfe,stroke:#6c5ce7,color:#000
    style R2 fill:#a29bfe,stroke:#6c5ce7,color:#000
    style R3 fill:#a29bfe,stroke:#6c5ce7,color:#000
```

---

## E.8 A Suggested 3-Month Plan

| Month | Focus | Activities |
|:------|:------|:-----------|
| **1** | Solidify foundations | Re-do all hands-on exercises; complete 2 Kaggle "Getting Started" competitions |
| **2** | Go deeper | Learn XGBoost; study regularisation; read ISLR chapters 5–8; start a Kaggle tabular competition |
| **3** | Explore deep learning | Take Fast.ai Part 1; build an image classifier; fine-tune a pre-trained model |

---

> **Remember:** The best way to learn ML is to **practise with real data**. Pick a dataset that interests you, define a question, and build a model end to end. You already have all the tools — go build something!
