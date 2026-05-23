# SHAPnLIME — Explainable AI for Intrusion Detection

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Research-orange)]()
[![Topic](https://img.shields.io/badge/Domain-Cybersecurity%20%7C%20XAI-red)]()

> **Interpretable machine learning pipeline for network intrusion detection — using SHAP and LIME to explain *why* a model flags traffic as malicious.**

---

## 🔍 Problem

Modern IDS solutions are black boxes. Security analysts get an alert but not a reason. This project answers the question: **"Which features drove this classification, and by how much?"**

---

## 🏗️ Architecture

```
Network Traffic (PCAP / CSV)
         │
         ▼
  [Feature Engineering]
  - Flow statistics, packet timing, protocol flags
         │
         ▼
  [ML Classifier]
  - Random Forest / XGBoost (trained on NSL-KDD / CIC-IDS-2018)
         │
         ▼
  [XAI Explainability Layer]
  ├── SHAP (global + local feature importance)
  └── LIME (local surrogate explanations per alert)
         │
         ▼
  [Analyst Dashboard]
  - Per-alert explanation report
  - Top contributing features
  - Decision confidence scores
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔬 **SHAP Analysis** | Global Shapley values — understand model behaviour across the dataset |
| 🧩 **LIME Explanations** | Per-prediction local explanations — why *this* packet was flagged |
| 📊 **Visualisations** | Force plots, summary plots, waterfall charts |
| 🗂️ **Multi-dataset support** | NSL-KDD, CIC-IDS-2017/2018, UNSW-NB15 |
| ⚡ **Fast inference** | Batch and real-time scoring modes |
| 📄 **Analyst reports** | Auto-generated PDF/HTML explanation reports per alert |

---

## 🚀 Quickstart

```bash
# Clone the repo
git clone https://github.com/ashvinauti/SHAPnLIME.git
cd SHAPnLIME

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run training + explanation pipeline
python src/train.py --dataset nsl-kdd
python src/explain.py --sample-id 42
```

---

## 📁 Project Structure

```
SHAPnLIME/
├── data/               # Raw and processed datasets (excluded from git)
├── notebooks/          # Exploratory analysis and visualisation
│   ├── 01_eda.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_xai_explanations.ipynb
├── src/
│   ├── features/       # Feature engineering pipeline
│   ├── models/         # Classifier training and evaluation
│   ├── explainability/ # SHAP + LIME wrappers
│   └── reports/        # Report generation utilities
├── tests/              # Unit and integration tests
├── requirements.txt
└── README.md
```

---

## 🧪 Datasets

| Dataset | Samples | Classes | Link |
|---|---|---|---|
| NSL-KDD | 125,973 | 5 | [UNB](https://www.unb.ca/cic/datasets/nsl.html) |
| CIC-IDS-2017 | 2.8M | 15 | [UNB](https://www.unb.ca/cic/datasets/ids-2017.html) |
| UNSW-NB15 | 257,673 | 10 | [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |

---

## 📈 Results (NSL-KDD)

| Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Random Forest | 99.1% | 0.991 | 0.990 | 0.992 |
| XGBoost | 99.3% | 0.993 | 0.991 | 0.995 |

SHAP identified **`dst_bytes`**, **`service`**, and **`flag`** as the top three features across attack categories.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/-Python%203.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/-XGBoost-FF6600)
![SHAP](https://img.shields.io/badge/-SHAP-007ACC)
![LIME](https://img.shields.io/badge/-LIME-00B050)
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter&logoColor=white)

---

## 🔗 Related Work

- 📄 Part of my MSc research (University of Hertfordshire — AI & Cybersecurity)
- 🔐 Integrated into the [AIPCS-SOC](https://github.com/ashvinauti/AIPCS-SOC) course module on AI-driven alert triage
- 👤 [LinkedIn](https://linkedin.com/in/ashvinauti) · [GitHub](https://github.com/ashvinauti)

---

## 📜 License

MIT © [Ashvin Auti](https://github.com/ashvinauti)
