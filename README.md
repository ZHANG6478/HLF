# HLF (High-Low Frequency Forecasting)

This repository contains the implementation of **HLF (High–Low Frequency Forecasting)**, a frequency-aware forecasting framework for time series prediction.

HLF automatically selects VMD hyperparameters, partitions intrinsic mode functions into high-/low-frequency components via joint energy–frequency analysis, and applies frequency-aligned forecasters to each branch before reconstructing the final forecast.

Some experimental results are omitted due to GitHub limitations; please contact the authors if needed.

---

## Repository Structure

```text
HLF-code/
├── Baseline/                  # Baseline models on Fashion Retail Sales dataset
├── Baseline(Exchange)/        # Baseline models on Exchange Rate dataset
├── HLF/                       # HLF method on Fashion Retail Sales dataset
├── HLF(Exchange)/             # HLF method on Exchange Rate dataset
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
