import pandas as pd
df = pd.read_csv("data/raw/accepted_2007_to_2018Q4.csv.gz", usecols=["funded_amnt","total_pymnt","total_rec_prncp","total_rec_int","total_rec_late_fee","recoveries","collection_recovery_fee"], low_memory=False)
df["sum"] = df["total_rec_prncp"] + df["total_rec_int"] + df["total_rec_late_fee"] + df["recoveries"]
diff = (df["total_pymnt"] - df["sum"]).abs()
print(f"cash verify: total_pymnt vs sum diff mean {diff.mean():.2f} max {diff.max():.2f} 95p {diff.quantile(0.95):.2f}")
print(f"holds if mean <1 and max <10: {diff.mean() < 1 and diff.max() < 10}")
# also check funded vs collection
print(df[["funded_amnt","collection_recovery_fee"]].describe().to_string())