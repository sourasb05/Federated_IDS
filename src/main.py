import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score, roc_curve, auc
)

import warnings
warnings.filterwarnings("ignore")

# our defined functions

import utils as utils
import models as models
import evaluate_model as evaluate_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    args = utils.parse_args()

    current_directory = os.path.dirname(os.path.abspath(__file__))  # /proj/.../Local_IDS/src
    project_root = os.path.dirname(current_directory)              # /proj/.../Local_IDS
    domains_path = os.path.join(project_root, 'attack_data') 
    
    
    domains = utils.create_domains(domains_path)

    train_domains_loader = {}
    test_domains_loader = {}
    per_domain_results = OrderedDict()


    for key, files in domains.items():
        train_domains_loader[key], test_domains_loader[key] = utils.load_data(domains_path, key, files, window_size=args.window_size, step_size=args.step_size, batch_size=args.batch_size)
          
    # each domain/key is a client now. We will train a federated model across all these clients/domains/keys.

    model = models.LSTMClassifier(input_dim=args.input_size, hidden_dim=args.hidden_size, output_dim=args.output_size, num_layers=args.num_layers, fc_hidden_dim=10).to(device)

    
    for iter in range(args.global_iters):
        
        print(f"\n--- Global Iteration {iter+1} / {args.global_iters} ---")

        local_models = []
        local_data_sizes = []

        for domain_key in domains.keys():
            print(f"\nTraining on domain: {domain_key}")

            local_model = models.LSTMClassifier(input_dim=args.input_size, hidden_dim=args.hidden_size, output_dim=args.output_size, num_layers=args.num_layers, fc_hidden_dim=10).to(device)
            local_model.load_state_dict(model.state_dict())  # Initialize with global model weights

            optimizer = optim.Adam(local_model.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss()

            train_loader = train_domains_loader[domain_key]
            test_loader = test_domains_loader[domain_key]

            local_model.train()
            for epoch in range(args.local_epochs):
                epoch_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                    optimizer.zero_grad()
                    outputs, _ = local_model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * X_batch.size(0)

                epoch_loss /= len(train_loader.dataset)
                print(f"Epoch {epoch+1}/{args.local_epochs}, Loss: {epoch_loss:.4f}")

            local_models.append(local_model.state_dict())
            local_data_sizes.append(len(train_loader.dataset))

        # Aggregate local models
        global_state_dict = model.state_dict()
        for key in global_state_dict.keys():
            global_state_dict[key] = sum(local_models[i][key] * (local_data_sizes[i] / sum(local_data_sizes)) for i in range(len(local_models)))
        model.load_state_dict(global_state_dict)

        for domain in domains.keys():
            metrics = evaluate_model.eval_global(model, test_domains_loader[domain], device)
            per_domain_results[domain] = {
                'n': metrics['n'], 'loss': None if metrics['loss'] is None else float(metrics['loss']),
                'acc': None if metrics['acc'] is None else float(metrics['acc']),
                'f1': None if metrics['f1'] is None else float(metrics['f1']),
                'auc': None if metrics['auc'] is None else float(metrics['auc']),
                'cm': None if metrics['cm'] is None else metrics['cm'].tolist(),
            }
            print(f"[{domain}] n={metrics['n']} | acc={metrics['acc']} | f1={metrics['f1']} | auc={metrics['auc']} | loss={metrics['loss']} | cm={metrics['cm']}")
        
    # aggregate (size-weighted) across domains for your own record
    total_n = sum((v['n'] or 0) for v in per_domain_results.values())
    auc_weighted = None
    if total_n > 0:
        acc_weighted = sum((v['acc'] or 0.0) * (v['n'] or 0) for v in per_domain_results.values()) / total_n
        f1_vals = [v for v in per_domain_results.values() if v['f1'] is not None and v['n']]
        f1_weighted = (sum(v['f1'] * v['n'] for v in f1_vals) / sum(v['n'] for v in f1_vals)) if f1_vals else None
        auc_vals = [v for v in per_domain_results.values() if v['auc'] is not None and v['n']]
        auc_weighted = (sum(v['auc'] * v['n'] for v in auc_vals) / sum(v['n'] for v in auc_vals)) if auc_vals else None
        loss_vals = [v for v in per_domain_results.values() if v['loss'] is not None and v['n']]
        loss_weighted = (sum(v['loss'] * v['n'] for v in loss_vals) / sum(v['n'] for v in loss_vals)) if loss_vals else None
        
    else:
        acc_weighted = f1_weighted = loss_weighted = None

    print("\n=== Aggregates (computed from per-domain) ===")
    print(f"Accuracy: {None if acc_weighted is None else f'{acc_weighted*100:.2f}%'}")
    print(f"F1: {None if f1_weighted is None else f'{f1_weighted:.4f}'}")
    print(f"AUC : {None if auc_weighted is None else f'{auc_weighted:.4f}'}")
    print(f"Loss : {None if loss_weighted is None else f'{loss_weighted:.4f}'}")

        
if __name__ == "__main__":

    main()