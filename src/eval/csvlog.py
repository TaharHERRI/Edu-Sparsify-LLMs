import os, pandas as pd

def append_row(csv_path, **row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=["setup","size_mb","sparsity","latency_ms","perplexity"]).to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
