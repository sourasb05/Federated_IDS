from http import server
import json
import os
from flask import json
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score, roc_curve, auc
)
import warnings
warnings.filterwarnings("ignore")
import sys

# our defined functions

from client import Client
from server import server
import utils as utils
import models as models
import evaluate_model as evaluate_model


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    args = utils.parse_args()

    current_directory = os.path.dirname(os.path.abspath(__file__))  # /proj/.../Local_IDS/src
    project_root = os.path.dirname(current_directory)              # /proj/.../Local_IDS
    domains_path = os.path.join(project_root, 'attack_data') 
    

    domains = utils.create_domains(domains_path)

    client_list = []

    tot_time_steps = args.tot_time_steps
    max_client_participants = 5
    overlap_p = args.overlap_p   # p=0 → disjoint, p=1 → all clients share all domains

    if args.attacktype:
        domain_keys = [key for key in domains.keys() if args.attacktype in key]
    else:
        domain_keys = list(domains.keys())

    print(f"{len(domain_keys)} domain_keys: {domain_keys}")

    client_distributions = utils.distribute_domains(
        domain_keys = domain_keys,
        n_clients   = max_client_participants,
        p           = overlap_p,
        seed        = args.seed,
    )

    # ── auto-compute input_size from data ──────────────────────────
    # input_size must equal window_size × n_raw_features.
    # Hardcoding it separately from window_size causes shape mismatches
    # when either value changes.  Peek at the first CSV to get n_features.
    first_domain_key  = domain_keys[0]
    first_file_name   = domains[first_domain_key][0]
    first_file_path   = os.path.join(domains_path, first_domain_key, first_file_name)
    _sample_df        = utils.load_csv(first_file_path)
    n_raw_features    = len([c for c in _sample_df.columns if c != 'label'])
    args.input_size   = args.window_size * n_raw_features
    args.n_raw_features = n_raw_features
    print(f"\nAuto-computed input_size: {args.window_size} (window) × "
          f"{n_raw_features} (features) = {args.input_size}")

    # input_dim = n_raw_features (per-step features), NOT window_size * n_raw_features.
    # The LSTM now sees (B, window_size, n_raw_features) — real temporal sequences.
    # num_layers=2 stacks two LSTM layers for deeper temporal abstraction.
    model = models.LSTMClassifier(input_dim=n_raw_features, hidden_dim=args.hidden_size, output_dim=args.output_size, num_layers=2, fc_hidden_dim=32).to(device)

    # Trigger the server execution based on the chosen algorithm
    if args.algorithm == 'fedavg':
        server(
            args=args, 
            model=model, 
            device=device, 
            domains_path=domains_path, 
            client_distributions=client_distributions, 
            max_client_participants=max_client_participants, 
            tot_time_steps=tot_time_steps, 
            project_root=project_root
        )

    elif args.algorithm == 'tabvae':
        from server_TabVAE import server_tabvae

        server_tabvae(
            args=args, 
            model=model, 
            device=device, 
            domains_path=domains_path, 
            client_distributions=client_distributions, 
            max_client_participants=max_client_participants, 
            tot_time_steps=tot_time_steps, 
            project_root=project_root
        )
    elif args.algorithm == 'tvae':
        from server_TVAE import server_tvae

        server_tvae(
            args=args,
            model=model,
            device=device,
            domains_path=domains_path,
            client_distributions=client_distributions,
            max_client_participants=max_client_participants,
            tot_time_steps=tot_time_steps,
            project_root=project_root
        )

    elif args.algorithm == 'Rvae':
        from server_RVAE import server_rvae

        server_rvae(
            args=args,
            model=model,
            device=device,
            domains_path=domains_path,
            client_distributions=client_distributions,
            max_client_participants=max_client_participants,
            tot_time_steps=tot_time_steps,
            project_root=project_root
        )

    elif args.algorithm == 'replay':
        from server_Replay import server_replay

        server_replay(
            args=args,
            model=model,
            device=device,
            domains_path=domains_path,
            client_distributions=client_distributions,
            max_client_participants=max_client_participants,
            tot_time_steps=tot_time_steps,
            project_root=project_root,
            buffer_size=500
        )

    elif args.algorithm == 'pcflta':
        from server_PCFLTA import server_pcflta

        server_pcflta(
            args=args,
            model=model,
            device=device,
            domains_path=domains_path,
            client_distributions=client_distributions,
            max_client_participants=max_client_participants,
            tot_time_steps=tot_time_steps,
            project_root=project_root,
            n_clusters=3
        )

    elif args.algorithm == 'gmm':
        from server_GMM import server_gmm

        server_gmm(
            args=args,
            model=model,
            device=device,
            domains_path=domains_path,
            client_distributions=client_distributions,
            max_client_participants=max_client_participants,
            tot_time_steps=tot_time_steps,
            project_root=project_root,
            n_components=10,
        )

    elif args.algorithm == 'timevae':
        from server_TimeVAE import server_timevae

        server_timevae(
            args=args,
            model=model,
            device=device,
            domains_path=domains_path,
            client_distributions=client_distributions,
            max_client_participants=max_client_participants,
            tot_time_steps=tot_time_steps,
            project_root=project_root,
        )

    elif args.algorithm == 'efl':
        from server_EFL import server_efl

        server_efl(
            args=args,
            device=device,
            domains_path=domains_path,
            client_distributions=client_distributions,
            max_client_participants=max_client_participants,
            tot_time_steps=tot_time_steps,
            project_root=project_root,
        )

if __name__ == "__main__":

    main()