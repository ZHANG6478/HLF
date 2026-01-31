# FDF (Frequency-aware Dual-model Forecasting)

This repository contains the implementation of **FDF (Frequency-aware Dual-model Forecasting)**, a frequency-aware dual-model forecasting framework for time series prediction.

FDF automatically selects VMD hyperparameters, splits modes into low/high-frequency components via joint energy–frequency analysis, and applies frequency-aligned predictors to each part before reconstructing the final forecast.

---

## Repository Structure

```text
FDF-code/
├── Baseline/                  # Baseline models on Fashion Retail Sales dataset
├── Baseline(Exchange)/        # Baseline models on Exchange Rate dataset
├── FDF/                       # FDF method on Fashion Retail Sales dataset
├── FDF(Exchange)/             # FDF method on Exchange Rate dataset
└── Preprocessing/             # Data preprocessing scripts
```

---

## Datasets

### 1) Fashion Retail Sales Dataset
Source (Kaggle):  
https://www.kaggle.com/datasets/atharvasoundankar/fashion-retail-sales

### 2) Exchange Rate Dataset
Source (HuggingFace):  
https://huggingface.co/datasets/thuml/Time-Series-Library/tree/main/exchange_rate

---

## Environment

Recommended:
- Python >= 3.8
- numpy
- pandas
- matplotlib
- scikit-learn
- scipy
- torch
- tensorflow
- vmdpy