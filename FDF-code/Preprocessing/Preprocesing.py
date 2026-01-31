import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
from sklearn.ensemble import IsolationForest


def init_env():
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.figsize"] = (12, 6)
    os.makedirs("results", exist_ok=True)


def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {os.path.abspath(path)}")


def load_and_clean_raw(file_path):
    check_file(file_path)

    excel_data = pd.read_excel(file_path, engine="openpyxl", sheet_name=None)
    df = excel_data[list(excel_data.keys())[0]]

    required_cols = ["Customer Reference ID", "Item Purchased", "Purchase Amount (USD)", "Date Purchase"]
    if not set(required_cols).issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols}")

    df["Date Purchase"] = pd.to_datetime(df["Date Purchase"], format="%d-%m-%Y", errors="coerce")
    df = df.dropna(subset=["Date Purchase"])

    df["Purchase Amount (USD)"] = pd.to_numeric(df["Purchase Amount (USD)"], errors="coerce")
    item_mean = df.groupby("Item Purchased")["Purchase Amount (USD)"].transform(lambda x: x.mean())
    df["Purchase Amount (USD)"] = df["Purchase Amount (USD)"].fillna(item_mean).fillna(df["Purchase Amount (USD)"].mean())

    return df


def category_time_series(df):
    category_map = {
        "Tops": ["T-shirt", "Polo Shirt", "Flannel Shirt", "Blouse", "Tunic",
                 "Tank Top", "Camisole", "Sweater", "Cardigan", "Hoodie", "Vest", "Kimono"],
        "Jumpsuits": ["Onesie", "Romper", "Jumpsuit"],
        "Jeans": ["Jeans", "Trousers", "Pants", "Shorts", "Leggings", "Overalls"],
        "Skirts": ["Dress", "Skirt"],
        "Outerwear": ["Jacket", "Blazer", "Trench Coat", "Coat", "Poncho", "Raincoat"],
        "Footwear": ["Loafers", "Sneakers", "Sandals", "Slippers", "Flip-Flops", "Boots"],
        "Accessories": ["Bowtie", "Tie", "Scarf", "Hat", "Sun Hat", "Sunglasses",
                        "Gloves", "Socks", "Belt"],
        "Bags": ["Handbag", "Backpack", "Wallet"],
        "Others": ["Pajamas", "Swimsuit", "Umbrella"],
    }
    item_to_cat = {item: cat for cat, items in category_map.items() for item in items}

    df["Item Purchased"] = df["Item Purchased"].str.title()
    df["category"] = df["Item Purchased"].map(item_to_cat)
    df = df[df["category"].notna()]

    time_series_dict = {}
    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat].copy()

        daily_sales = cat_df.groupby("Date Purchase")["Purchase Amount (USD)"].sum().reset_index()

        start, end = cat_df["Date Purchase"].min(), cat_df["Date Purchase"].max()
        all_dates = pd.date_range(start, end, freq="D")
        continuous_df = pd.DataFrame({"Date Purchase": all_dates})

        continuous_df = pd.merge(continuous_df, daily_sales, on="Date Purchase", how="left").fillna(0)
        continuous_df.columns = ["date", "daily_amount_usd"]
        time_series_dict[cat] = continuous_df

    return time_series_dict


def optimize_time_series(cat, ts_df):
    def fix_zero(series):
        fixed = series.copy()
        for i in fixed[fixed == 0].index:
            window = fixed[max(0, i - 3):min(len(fixed), i + 4)]
            fixed[i] = window[window != 0].mean() if len(window[window != 0]) > 0 else 0
        return fixed

    ts_df["zero_fixed"] = fix_zero(ts_df["daily_amount_usd"])

    ts_df["roll_mean"] = ts_df["zero_fixed"].rolling(21, center=True).mean().fillna(0)
    ts_df["diff"] = ts_df["zero_fixed"].diff().fillna(0)
    ts_df["roll_std"] = ts_df["zero_fixed"].rolling(21, center=True).std().fillna(0)

    features = ts_df[["roll_mean", "diff", "roll_std"]].to_numpy()
    model = IsolationForest(contamination=0.08, random_state=42)
    ts_df["is_noise"] = (model.fit_predict(features) == -1)

    ts_df["noise_fixed"] = ts_df["zero_fixed"]
    for i in ts_df[ts_df["is_noise"]].index:
        ts_df.loc[i, "noise_fixed"] = ts_df.loc[max(0, i - 3):min(len(ts_df) - 1, i + 3), "zero_fixed"].mean()

    ts_df["smoothed_5d"] = ts_df["noise_fixed"].rolling(3, center=True, min_periods=1).mean().ffill().bfill()
    return ts_df


def save_and_plot(cat, optimized_df):
    safe_cat = str(cat).replace("/", "_").replace("\\", "_")

    out_xlsx = f"results/{safe_cat}_timeseries.xlsx"
    out_png = f"results/{safe_cat}_trend.png"

    export_df = optimized_df[["date", "zero_fixed", "is_noise", "smoothed_5d"]].copy()
    export_df = export_df.rename(columns={
        "zero_fixed": "Sales(Pre)",
        "smoothed_5d": "Sales",
    })
    export_df.to_excel(out_xlsx, index=False, engine="openpyxl")

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=300)

    ax.plot(
        optimized_df["date"], optimized_df["daily_amount_usd"],
        linestyle="--", linewidth=1.0, alpha=0.6, label="Raw"
    )
    ax.plot(
        optimized_df["date"], optimized_df["smoothed_5d"],
        linestyle="-", linewidth=1.6, label="Sales"
    )

    noise_df = optimized_df[optimized_df["is_noise"]]
    if len(noise_df) > 0:
        ax.scatter(
            noise_df["date"], noise_df["smoothed_5d"],
            s=14, marker="o", alpha=0.8, label="Noise"
        )

    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))

    ax.set_title(f"{cat}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales (USD)")

    ax.grid(True, which="major", linewidth=0.5, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, loc="best")
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)



def main():
    init_env()
    raw_file = "Fashion_Retail_Sales.xlsx"
    raw_cleaned = load_and_clean_raw(raw_file)
    category_ts = category_time_series(raw_cleaned)

    for cat, ts_df in category_ts.items():
        optimized = optimize_time_series(cat, ts_df)
        save_and_plot(cat, optimized)

    print("Done. Results saved to: results/")


if __name__ == "__main__":
    main()
