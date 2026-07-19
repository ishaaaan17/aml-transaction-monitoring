import pandas as pd
from data_utils import load_transactions

df = load_transactions(path="../data/HI-Small_Trans.csv")

print("Unique Account values:", df["Account"].nunique())
print("Unique (Account, From Bank) pairs:", df.groupby(["Account", "From Bank"]).ngroups)
print()

suspicious = df[df["Account"] == "8000EBD30"]
print("Rows with this Account string:", len(suspicious))
print("Unique From Bank values for this Account string:", suspicious["From Bank"].nunique())
print(suspicious["From Bank"].value_counts().head())

top_account = df["Account"].value_counts().idxmax()
top_count = df["Account"].value_counts().max()
print(f"Top account: {top_account}, transaction count: {top_count}")

top_rows = df[df["Account"] == top_account]
print("From Bank values for this account:", top_rows["From Bank"].unique())
print("Is Laundering distribution for this account:")
print(top_rows["Is Laundering"].value_counts())
print()
print("Sample timestamps:")
print(top_rows["Timestamp"].sort_values().head(10))
print(top_rows["Timestamp"].sort_values().tail(10))