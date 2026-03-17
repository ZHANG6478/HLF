# A-HLFDM (Adaptive VMD-based High–Low Frequency Differentiated Modeling)

This repository provides the implementation of **A-HLFDM**, a hybrid soft-computing framework for forecasting highly volatile and non-stationary time series with mixed-frequency dynamics.

A-HLFDM combines causality-preserving preprocessing, adaptive VMD-based decomposition, and frequency-aligned modeling to improve forecasting accuracy, stability, and interpretability. It automatically selects VMD hyperparameters in a data-driven manner and partitions intrinsic mode functions (IMFs) into low- and high-frequency components using joint energy–frequency analysis. Frequency-aligned predictors are then applied to each component, and final forecasts are obtained through linear reconstruction.

Some experimental results are omitted due to GitHub limitations; please contact the authors if needed.

---

## Method Overview

A-HLFDM follows a five-stage pipeline:

1. **Data Preprocessing**  
   The raw series is purified using a causality-preserving procedure. Temporal features are constructed from historical data, anomalies are detected via Isolation Forest, and corrected using a strictly causal strategy.

2. **Adaptive VMD Decomposition**  
   The series is decomposed into intrinsic mode functions (IMFs) using VMD, with key hyperparameters selected automatically through data-driven optimization.

3. **High–Low Frequency Partitioning**  
   IMFs are sorted by frequency and partitioned into low- and high-frequency groups based on a cumulative energy criterion.

4. **Frequency-Aligned Forecasting**  
   Low-frequency components are modeled using regression-based approaches, while high-frequency components are modeled using dependency-aware models under a rolling one-step-ahead scheme.

5. **Forecast Reconstruction**  
   Final predictions are obtained by linearly aggregating component-wise forecasts.

This pipeline enables effective modeling of mixed-frequency dynamics for accurate, stable, and interpretable forecasting.

---

## Repository Structure

```text
A-HLFDM-code/
├── Baseline/                  # Baseline models on Fashion Retail Sales dataset
├── Baseline(Exchange)/        # Baseline models on Exchange Rate dataset
├── A-HLFDM/                   # A-HLFDM method on Fashion Retail Sales dataset
├── A-HLFDM(Exchange)/         # A-HLFDM method on Exchange Rate dataset
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
