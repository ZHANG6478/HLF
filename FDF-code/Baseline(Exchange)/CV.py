import numpy as np
import pandas as pd

df = pd.read_csv("exchange_rate.csv")

first_col = str(df.columns[0]).lower()
data = df.iloc[:, 1:] if "date" in first_col else df

stds = data.std()
means = data.mean()

eps = 1e-12
cv = stds / (means.abs() + eps)

cv_sorted = cv.sort_values(ascending=False)
best_col = cv_sorted.index[0]

print("CV by column (desc):")
print(cv_sorted)

print(f"\nBest column: {best_col}")
print(f"std={stds[best_col]} mean={means[best_col]} CV={cv[best_col]}")
