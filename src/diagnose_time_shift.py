import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from data_utils import load_transactions

df = load_transactions(path="../data/HI-Small_Trans.csv")

daily = df.groupby(df["Timestamp"].dt.date)["Is Laundering"].agg(["sum", "count", "mean"])
daily.columns = ["laundering_count", "total_count", "laundering_rate"]
daily["laundering_rate_pct"] = daily["laundering_rate"] * 100

print(daily[["laundering_count", "total_count", "laundering_rate_pct"]])