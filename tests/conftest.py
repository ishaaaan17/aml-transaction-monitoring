import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
for p in (SRC, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def toy_transactions():
    """A tiny IBM-AML-shaped frame: 2 accounts x 5 hourly transactions."""
    rows = []
    ts = pd.date_range("2022-09-01", periods=5, freq="h")
    for acct, bank in [("A1", 10), ("B2", 20)]:
        for i, t in enumerate(ts):
            rows.append({
                "Timestamp": t,
                "From Bank": bank,
                "Account": acct,
                "To Bank": bank + 1,
                "Account.1": "C9",
                "Amount Received": 100.0 * (i + 1),
                "Receiving Currency": "USD",
                "Amount Paid": 100.0 * (i + 1),
                "Payment Currency": "USD",
                "Payment Format": "ACH" if i == 4 else "Wire",
                "Is Laundering": 1 if (acct == "A1" and i == 4) else 0,
            })
    return pd.DataFrame(rows)
