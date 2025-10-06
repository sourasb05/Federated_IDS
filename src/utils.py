import os
import random
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
import logging
import argparse
import sys
import re
import wandb

def safe_minmax_normalize(df, global_min, global_max, label_col="label"):
    feat_cols = [c for c in df.columns if c != label_col]
    denom = (global_max - global_min).replace(0, 1)  # avoid div/0
    out = df.copy()
    out[feat_cols] = (out[feat_cols] - global_min) / denom
    out[feat_cols] = out[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out

def seq_maker(df, sequence_length=10, label_col="label"):
    df_feat = df.drop(columns=[label_col])
    labels = df[label_col].astype(int).values

    attack_idxs = np.where(labels == 1)[0]
    if len(attack_idxs) == 0:
        start_attack = len(labels) + sequence_length   # all zeros
    else:
        start_attack = max(0, attack_idxs[0] - sequence_length)

    sequences = []
    for i in range(len(df_feat) - sequence_length):
        sequences.append(df_feat.iloc[i:i+sequence_length].values.flatten())

    if not sequences:
        return pd.DataFrame(columns=[*range(df_feat.shape[1]*sequence_length), "label"])

    seq_df = pd.DataFrame(sequences)
    zeros = [0] * min(start_attack, len(seq_df))
    ones  = [1] * (len(seq_df) - len(zeros))
    seq_df["label"] = zeros + ones
    return seq_df

def extract_index(path):
    # pull the number between ..._ and _60_sec.csv
    m = re.search(r"_(\d+)_60_sec\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else 10**9  # push unknowns to end
DROP_COLS = ["Unnamed: 0"]
def load_csv(path):
    
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="ignore")
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
    assert "label" in df.columns, f"'label' column missing in {os.path.basename(path)}"
    return df


def save_results_as_json(results, filename, save_folder="results"):
    os.makedirs(save_folder, exist_ok=True)
    filepath = os.path.join(save_folder, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {filepath}")


# Function to create sliding windows
def create_sliding_windows(X, y, window_size, step_size):
    sequences, labels = [], []
    for i in range(0, len(X) - window_size, step_size):
        sequences.append(X[i:i + window_size])
        labels.append(y[i + window_size - 1])  # Label is the last time step in the window
    return np.array(sequences), np.array(labels)


def load_data(domain_path, key, domain_dataset, window_size=10, step_size=3, batch_size=128):

    files = sorted(domain_dataset, key=extract_index)[:20]  # ensure exactly 20, ordered

    DROP_COLS = ["Unnamed: 0"]
    
    random.seed(42)
    random.shuffle(files)
    train_files_wo_path = files[:16]
    test_files_wo_path  = files[16:20]


    train_files = [domain_path + "/" + key  + "/" + f for f in train_files_wo_path]
    test_files = [domain_path + "/" + key + "/" + f for f in test_files_wo_path]


    

    # print("Train files:", train_files) 
    # print("Test  files:", test_files)

    # Load your data (assuming it's already loaded as `data`)
    # -----------------------
    # Load all, compute train-only global min/max (excluding 'label')
    # -----------------------
    train_dfs = [load_csv(p) for p in train_files]
    test_dfs  = [load_csv(p) for p in test_files]

    feat_cols = [c for c in train_dfs[0].columns if c != "label"]
    train_feat_mins = [df[feat_cols].min(axis=0) for df in train_dfs]
    train_feat_maxs = [df[feat_cols].max(axis=0) for df in train_dfs]
    global_min = pd.concat(train_feat_mins, axis=1).min(axis=1)
    global_max = pd.concat(train_feat_maxs, axis=1).max(axis=1)

    # -----------------------
    # Normalize using train stats
    # -----------------------
    norm_train = [safe_minmax_normalize(df, global_min, global_max, "label") for df in train_dfs]
    norm_test  = [safe_minmax_normalize(df, global_min, global_max, "label") for df in test_dfs]

    # -----------------------
    # Sequence-ify and concat
    # -----------------------
    seq_train_parts = [seq_maker(df, window_size, "label") for df in norm_train]
    seq_test_parts  = [seq_maker(df, window_size, "label") for df in norm_test]

    seq_train_parts = [df for df in seq_train_parts if not df.empty]
    seq_test_parts  = [df for df in seq_test_parts if not df.empty]

    seq_train = pd.concat(seq_train_parts, ignore_index=True)
    seq_test  = pd.concat(seq_test_parts,  ignore_index=True)


    # -----------------------
    # Tensors & Dataloaders
    # -----------------------
    X_train = torch.tensor(seq_train.iloc[:, :-1].values, dtype=torch.float32)
    y_train = torch.tensor(seq_train.iloc[:,  -1].values.astype(int), dtype=torch.long)
    X_test  = torch.tensor(seq_test.iloc[:,  :-1].values, dtype=torch.float32)
    y_test  = torch.tensor(seq_test.iloc[:,   -1].values.astype(int), dtype=torch.long)

    X_train = torch.nan_to_num(X_train, nan=0.0)
    X_test  = torch.nan_to_num(X_test,  nan=0.0)

    feature_dim = X_train.shape[1]  # (#features * SEQUENCE_LENGTH)
    X_train = X_train.view(-1, 1, feature_dim)
    X_test  = X_test.view(-1, 1, feature_dim)
    
    # print(f"After view: X_train={X_train.shape}, X_test={X_test.shape}")


    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test,  y_test)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  len(test_dataset), shuffle=False)


    # print("X_train:", X_train.shape, "y_train:", y_train.shape)
    # print("X_test :", X_test.shape,  "y_test :", y_test.shape)

    return train_loader, test_loader

def create_domains(domains_path):

        # Iterate over each item in the source folder.
    domains = {}
    for domain  in os.listdir(domains_path):
        
        domain_path = os.path.join(domains_path, domain)
        
        if os.path.isdir(domain_path):
            files = sorted(os.listdir(domain_path))
            selected_files = files
           # Use the subfolder name as the dictionary key and its list of file contents as the value.
            domains[domain] = selected_files
    logging.info(f"Domains found: {domains.keys()}")
    logging.info(f"Number of domains found: {len(domains)}")

    # for key, value_list in domains.items():
    #     print(f"key : {key} value : {value_list} num_element : {len(value_list)}")
    return domains

def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

def confidence_from_logits(logits: torch.Tensor):
    probs = torch.softmax(logits, dim=1)     # (B, C)
    confs, preds = probs.max(dim=1)          # (B,), (B,)
    return probs, preds, confs



def _json_safe(obj):
    """Recursively convert NumPy/PyTorch/Pandas/sets/tuples into JSON-serializable types."""
    import numpy as np
    import torch

    # NumPy scalars
    if isinstance(obj, np.generic):
        return obj.item()

    # NumPy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # PyTorch tensors
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    # Pandas types
    try:
        import pandas as pd
        if isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="list")
    except Exception:
        pass

    # Builtins that need conversion
    if isinstance(obj, (set, tuple)):
        return [_json_safe(x) for x in obj]

    # Dicts / lists: recurse
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]

    # Everything else: leave as is (int/float/str/bool/None)
    return obj


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning on IDS Domains")

    # Model hyperparameters
    parser.add_argument('--input_size', type=int, default=140,
                        help='Number of input features per time step (e.g., 14 for IDS dataset)')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden size of LSTM')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of LSTM layers')
    parser.add_argument('--output_size', type=int, default=2,
                        help='Number of output classes (e.g., 2 for benign vs attack)')

    # Data processing hyperparameters
    parser.add_argument('--window_size', type=int, default=10,
                        help='Sliding window size for sequence generation')
    parser.add_argument('--step_size', type=int, default=3,
                        help='Step size for sliding window')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for DataLoader')

    # Federated learning hyperparameters
    parser.add_argument('--global_iters', type=int, default=5,
                        help='Number of global communication rounds')
    parser.add_argument('--local_epochs', type=int, default=5,
                        help='Number of local epochs per domain/client')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for local optimizers')

    # Paths and environment
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results (JSON, logs, etc.)')

    args = parser.parse_args()
    return args
