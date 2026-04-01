# server_PCFLTA.py
#
# PCFL-TA: Trajectory-Aligned Clustered Federated Learning server.
#
# Algorithm 1 (full server logic):
#   Initialize: Global Backbone Φ₀, Cluster Heads Ψ₁...k
#   for each round t = 1,...,T:
#     for each client i in parallel:
#       θᵢ ← LocalUpdate(Φₜ, Ψ_cluster)        (client side)
#       V⃗ᵢ ← mean(H_atk) − mean(H_norm)        (client side)
#       Upload {Δθᵢ, V⃗ᵢ} to Server
#     Sᵢⱼ ← CosineSimilarity(V⃗ᵢ, V⃗ⱼ)          (server: build similarity matrix)
#     C₁,...,Cₖ ← K-Medoids(S)                  (server: cluster by attack type)
#     Φₜ₊₁ ← GlobalAvg(Δφ)                      (server: average all backbone updates)
#     Ψₖ,ₜ₊₁ ← ClusterAvg(Δψ, i ∈ Cₖ)          (server: average heads within cluster)

import os
import copy

import numpy as np
import torch
import torch.nn as nn

from client_PCFLTA import ClientPCFLTA
from models_pcflta import LSTMBackbone, ClassifierHead
from utils import save_results_as_json


# ═══════════════════════════════════════════════════════════════
# UTILITY: cosine similarity matrix
# ═══════════════════════════════════════════════════════════════
def _cosine_similarity_matrix(vectors: list) -> np.ndarray:
    """
    Build an (n × n) cosine similarity matrix from a list of
    1-D numpy arrays (the client attack-signature vectors V⃗ᵢ).

    S[i,j] ∈ [-1, 1].  Values close to 1 → similar attack types.
    """
    V    = np.stack(vectors).astype(np.float64)          # (n, d)
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
    V_n  = V / norms
    return V_n @ V_n.T                                    # (n, n)


# ═══════════════════════════════════════════════════════════════
# UTILITY: K-Medoids (PAM — Partitioning Around Medoids)
# ═══════════════════════════════════════════════════════════════
def _kmedoids(dist_matrix: np.ndarray, k: int,
              max_iter: int = 100,
              random_state: int = 42) -> tuple:
    """
    Simple PAM K-Medoids on a pre-computed distance matrix.

    Args:
        dist_matrix  : (n, n) symmetric non-negative distance matrix.
                       Use 1 - cosine_similarity to convert from S.
        k            : number of clusters (≤ n)
        max_iter     : maximum PAM iterations
        random_state : for reproducible medoid initialisation

    Returns:
        labels  : (n,) int array — cluster index for each point
        medoids : list[int]      — indices of medoid points
    """
    n   = len(dist_matrix)
    k   = min(k, n)
    rng = np.random.RandomState(random_state)

    # ── initialise medoids randomly ───────────────
    medoids = rng.choice(n, size=k, replace=False).tolist()

    for _ in range(max_iter):
        # assign each point to the nearest medoid
        labels = np.argmin(dist_matrix[:, medoids], axis=1)

        # update each medoid to the point in the cluster that
        # minimises the total within-cluster distance
        new_medoids = []
        for c in range(k):
            cluster_idx = np.where(labels == c)[0]
            if len(cluster_idx) == 0:
                new_medoids.append(medoids[c])    # keep old medoid
                continue
            sub = dist_matrix[np.ix_(cluster_idx, cluster_idx)]
            best_local = int(np.argmin(sub.sum(axis=1)))
            new_medoids.append(int(cluster_idx[best_local]))

        if sorted(new_medoids) == sorted(medoids):
            break
        medoids = new_medoids

    labels = np.argmin(dist_matrix[:, medoids], axis=1)
    return labels, medoids


# ═══════════════════════════════════════════════════════════════
# UTILITY: average a list of state_dicts
# ═══════════════════════════════════════════════════════════════
def _avg_state_dicts(state_dicts: list) -> dict:
    """FedAvg / ClusterAvg: element-wise mean of state dicts."""
    avg = copy.deepcopy(state_dicts[0])
    n   = len(state_dicts)
    for key in avg:
        avg[key] = sum(sd[key] for sd in state_dicts) / n
    return avg


# ═══════════════════════════════════════════════════════════════
# MAIN SERVER FUNCTION
# ═══════════════════════════════════════════════════════════════
def server_pcflta(args, model, device, domains_path,
                  client_distributions, max_client_participants,
                  tot_time_steps, project_root,
                  n_clusters: int = 3):
    """
    PCFL-TA server: clustered federated learning with attack-trajectory
    signatures for client grouping.

    n_clusters : number of expert cluster heads (≤ n_clients).
                 Automatically capped to max_client_participants.
    """
    n_clusters = min(n_clusters, max_client_participants)
    hidden_dim    = args.hidden_size
    fc_hidden_dim = 10            # matches LSTMClassifier default
    input_dim     = args.input_size
    output_dim    = args.output_size

    print(
        f"\n--- Starting PCFL-TA  "
        f"(n_clusters={n_clusters}, "
        f"clients={max_client_participants}) ---"
    )

    models_dir = os.path.join(project_root, 'saved_models_pcflta')
    os.makedirs(models_dir, exist_ok=True)

    # ── initialise global backbone Φ₀ ────────────
    backbone = LSTMBackbone(input_dim, hidden_dim,
                            args.num_layers).to(device)

    # ── initialise k cluster heads Ψ₁...k ────────
    heads = [
        ClassifierHead(hidden_dim, fc_hidden_dim, output_dim).to(device)
        for _ in range(n_clusters)
    ]

    # ── initialise clients ────────────────────────
    client_list = []
    for i in range(max_client_participants):
        c = ClientPCFLTA(
            client_id        = i,
            args             = args,
            domain_path      = domains_path,
            assigned_domains = client_distributions[i],
            device           = device,
            hidden_dim       = hidden_dim,
            fc_hidden_dim    = fc_hidden_dim
        )
        client_list.append(c)

    print(f"Initialized {len(client_list)} ClientPCFLTA clients.")

    # ── cluster assignment: round 0 → all clients in cluster 0
    cluster_assignments = {c.client_id: 0 for c in client_list}

    # ── metric trackers ───────────────────────────
    loss_tracker         = {}
    acc_tracker          = {}
    f1_tracker           = {}
    cluster_history      = {}        # {time_step: {client_id: cluster_id}}
    task_accuracy_matrix = {t: {} for t in range(tot_time_steps)}
    bwt_tracker          = {}

    ACCURACY_THRESHOLD = 0.95

    # ════════════════════════════════════════════════════════
    # MAIN LOOP  (Algorithm 1, line 2: for t = 1,...,T)
    # ════════════════════════════════════════════════════════
    for t in range(tot_time_steps):
        print(f"\n{'='*50}")
        print(f"--- Time Step {t+1} / {tot_time_steps} ---")
        print(f"--- Cluster assignments: {cluster_assignments} ---")
        print(f"{'='*50}")

        loss_tracker[t]    = []
        acc_tracker[t]     = []
        f1_tracker[t]      = []
        cluster_history[t] = dict(cluster_assignments)

        best_acc_so_far = -1.0
        active_clients  = client_list

        # ── dynamic client selection ──────────────
        if t > 0:
            prev_model_path = os.path.join(
                models_dir, f"backbone_ts_{t}.pth"
            )
            backbone.load_state_dict(
                torch.load(prev_model_path, map_location=device)
            )
            for k in range(n_clusters):
                head_path = os.path.join(
                    models_dir, f"head_{k}_ts_{t}.pth"
                )
                heads[k].load_state_dict(
                    torch.load(head_path, map_location=device)
                )

            active_clients = []
            print(f"\nEvaluating clients for time step {t+1} ...")
            for c in client_list:
                k = cluster_assignments[c.client_id]
                _, initial_acc, _ = c.evaluate_global_model(
                    backbone.state_dict(),
                    heads[k].state_dict(),
                    time_step=t
                )
                print(
                    f"  Client {c.client_id} (cluster {k}) "
                    f"initial acc: {initial_acc:.4f}"
                )
                if initial_acc < ACCURACY_THRESHOLD:
                    active_clients.append(c)

            print(
                f"\n-> {len(active_clients)} clients selected "
                f"(acc < {ACCURACY_THRESHOLD*100:.0f}%)."
            )

            if len(active_clients) == 0:
                _save_checkpoint(backbone, heads, models_dir, t + 1,
                                 n_clusters)
                continue

        # ── global iteration loop ─────────────────
        for iter_idx in range(args.global_iters):
            print(
                f"\n--- Global Iter {iter_idx+1} / {args.global_iters} ---"
            )

            backbone_updates = []          # one per active client
            head_updates     = {}          # {client_id: head_state_dict}
            signatures       = {}          # {client_id: V⃗ᵢ}

            # ── Algorithm 1 lines 3-8: local updates ─
            for c in active_clients:
                k = cluster_assignments[c.client_id]
                bk_state, hd_state, sig = c.train(
                    backbone_state = backbone.state_dict(),
                    head_state     = heads[k].state_dict(),
                    cluster_id     = k,
                    time_step      = t
                )
                backbone_updates.append(bk_state)
                head_updates[c.client_id] = hd_state
                signatures[c.client_id]   = sig

            # ── Algorithm 1 line 9-10: K-Medoids ────
            # Build cosine similarity matrix S on signatures V⃗ᵢ
            # then convert to distance D = 1 - S for K-Medoids.
            if len(signatures) >= 2:
                client_ids = sorted(signatures.keys())
                sig_list   = [signatures[i] for i in client_ids]
                S = _cosine_similarity_matrix(sig_list)     # (n, n)
                D = np.clip(1.0 - S, 0.0, 2.0)             # distance
                labels, medoids = _kmedoids(D, k=n_clusters)

                print("\n  [Server] Cosine Similarity Matrix S:")
                for row in S:
                    print("  " + "  ".join(f"{v:+.3f}" for v in row))
                print(
                    f"  [Server] K-Medoids → "
                    f"labels={labels.tolist()}  medoids={medoids}"
                )

                for j, cid in enumerate(client_ids):
                    cluster_assignments[cid] = int(labels[j])

                print(
                    f"  [Server] New cluster assignments: "
                    f"{cluster_assignments}"
                )
            else:
                # Only 1 active client — no re-clustering needed
                print("  [Server] Only 1 active client; skipping K-Medoids.")

            # ── Algorithm 1 line 11: GlobalAvg backbone ─
            avg_backbone = _avg_state_dicts(backbone_updates)
            backbone.load_state_dict(avg_backbone)

            # ── Algorithm 1 line 12: ClusterAvg heads ───
            for k in range(n_clusters):
                # clients whose UPDATED assignment is k
                clients_in_k = [
                    cid for cid, cl in cluster_assignments.items()
                    if cl == k and cid in head_updates
                ]
                if clients_in_k:
                    avg_head = _avg_state_dicts(
                        [head_updates[cid] for cid in clients_in_k]
                    )
                    heads[k].load_state_dict(avg_head)
                    print(
                        f"  [Server] Head {k} updated from "
                        f"clients {clients_in_k}"
                    )

            # ── evaluate ALL clients ─────────────────
            print(
                f"\n--- Evaluating ALL clients at Time Step {t+1} ---"
            )
            total_loss = 0.0
            total_acc  = 0.0
            total_f1   = 0.0

            for c in client_list:
                k = cluster_assignments[c.client_id]
                eval_loss, acc, f1 = c.evaluate_global_model(
                    backbone.state_dict(),
                    heads[k].state_dict(),
                    time_step=t
                )
                total_loss += eval_loss
                total_acc  += acc
                total_f1   += f1

            avg_loss = total_loss / len(client_list)
            avg_acc  = total_acc  / len(client_list)
            avg_f1   = total_f1   / len(client_list)

            loss_tracker[t].append(avg_loss)
            acc_tracker[t].append(avg_acc)
            f1_tracker[t].append(avg_f1)

            print(
                f"Avg Accuracy across ALL clients: {avg_acc:.4f}"
            )

            # ── save best checkpoint ─────────────────
            if avg_acc > best_acc_so_far:
                best_acc_so_far = avg_acc
                _save_checkpoint(backbone, heads, models_dir, t + 1,
                                 n_clusters)
                print(
                    f"  --> [Saved] New best checkpoint "
                    f"for Time Step {t+1}  acc={avg_acc:.4f}"
                )

        # ── BWT evaluation ────────────────────────
        print(
            f"\n--- Evaluating Forgetting (BWT) "
            f"at end of Time Step {t+1} ---"
        )
        _load_checkpoint(backbone, heads, models_dir, t + 1, n_clusters,
                         device)

        for past_t in range(t + 1):
            past_total_acc = 0.0
            for c in client_list:
                k = cluster_assignments[c.client_id]
                _, acc, _ = c.evaluate_global_model(
                    backbone.state_dict(),
                    heads[k].state_dict(),
                    time_step=past_t
                )
                past_total_acc += acc
            avg_past_acc = past_total_acc / len(client_list)
            task_accuracy_matrix[t][past_t] = avg_past_acc
            print(
                f"  Performance on Time Step {past_t+1}: "
                f"{avg_past_acc:.4f}"
            )

        if t == 0:
            bwt_tracker[t] = 0.0
        else:
            bwt_sum = sum(
                task_accuracy_matrix[t][p] - task_accuracy_matrix[p][p]
                for p in range(t)
            )
            bwt_tracker[t] = bwt_sum / t
            print(
                f"\n--> BWT for Time Step {t+1}: {bwt_tracker[t]:.4f}"
            )
            if bwt_tracker[t] < 0:
                print(
                    f"    (Forgot "
                    f"{abs(bwt_tracker[t])*100:.2f}% on average)"
                )

    # ── save results ──────────────────────────────
    print("\n--- PCFL-TA Training Complete ---")

    results_to_save = {
        "loss"                : loss_tracker,
        "accuracy"            : acc_tracker,
        "f1_score"            : f1_tracker,
        "task_accuracy_matrix": task_accuracy_matrix,
        "bwt"                 : bwt_tracker,
        "cluster_history"     : cluster_history,   # cluster assignments per round
        "n_clusters"          : n_clusters,
    }

    results_folder = os.path.join(project_root, 'results')
    save_results_as_json(
        'pcflta_metrics.json',
        results_to_save,
        project_root,
        results_folder
    )


# ═══════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ═══════════════════════════════════════════════════════════════
def _save_checkpoint(backbone, heads, models_dir, time_step, n_clusters):
    torch.save(backbone.state_dict(),
               os.path.join(models_dir, f"backbone_ts_{time_step}.pth"))
    for k, head in enumerate(heads):
        torch.save(head.state_dict(),
                   os.path.join(models_dir, f"head_{k}_ts_{time_step}.pth"))


def _load_checkpoint(backbone, heads, models_dir, time_step, n_clusters,
                     device):
    backbone.load_state_dict(
        torch.load(
            os.path.join(models_dir, f"backbone_ts_{time_step}.pth"),
            map_location=device
        )
    )
    for k, head in enumerate(heads):
        head.load_state_dict(
            torch.load(
                os.path.join(models_dir, f"head_{k}_ts_{time_step}.pth"),
                map_location=device
            )
        )
