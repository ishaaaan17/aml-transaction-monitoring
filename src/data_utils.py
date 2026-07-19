import pandas as pd

DTYPE_MAP = {
    "From Bank": "int32",
    "Account": "category",
    "To Bank": "int32",
    "Account.1": "category",
    "Amount Received": "float32",
    "Receiving Currency": "category",
    "Amount Paid": "float32",
    "Payment Currency": "category",
    "Payment Format": "category",
    "Is Laundering": "int8",
}


def load_transactions(path="../data/HI-Small_Trans.csv"):
    """Load the IBM AML HI-Small transaction CSV with memory-optimized dtypes."""
    df = pd.read_csv(
        path,
        dtype=DTYPE_MAP,
        parse_dates=["Timestamp"],
        date_format="%Y/%m/%d %H:%M",
    )
    return df


if __name__ == "__main__":
    df = load_transactions()
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    print(f"Laundering rate: {df['Is Laundering'].mean() * 100:.4f}%")