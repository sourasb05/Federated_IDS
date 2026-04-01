# server_Replay.py
#
# FedAvg server using ClientReplay clients (experience replay).
#
# Unlike generative replay (TVAE / RVAE), each client here stores a
# fixed buffer of REAL samples from every past domain and replays them
# verbatim during local training on a new domain.
#
# Aggregation logic is identical to server_RVAE.py and server_TVAE.py.
# The only differences:
#   - Uses ClientReplay instead of ClientRVAE / ClientTVAE
#   - Saves results to 'replay_metrics.json'
#   - Collects buffer_quality (KS results) from c.buffer_eval_results

import os
import torch
from client_Replay import ClientReplay
from utils import save_results_as_json


def server_replay(args, model, device, domains_path,
                  client_distributions, max_client_participants,
                  tot_time_steps, project_root,
                  buffer_size: int = 500):
    """
    Executes FedAvg server logic with ClientReplay clients.

    Each client privately maintains one real-sample buffer per domain.
    Buffers are never shared or aggregated — they stay on each client.

    buffer_size : number of real samples stored per domain per client.
    """
    print("\n--- Starting FedAvg with Experience Replay (real samples) ---")

    models_dir = os.path.join(project_root, 'saved_models_replay')
    os.makedirs(models_dir, exist_ok=True)

    # ── initialise clients ───────────────────────
    client_list = []
    for i in range(max_client_participants):
        c = ClientReplay(
            client_id        = i,
            args             = args,
            domain_path      = domains_path,
            assigned_domains = client_distributions[i],
            device           = device,
            model            = model,
            buffer_size      = buffer_size
        )
        client_list.append(c)

    print(f"\nInitialized {len(client_list)} ClientReplay clients "
          f"(buffer_size={buffer_size} per domain).")

    # ── metric trackers ──────────────────────────
    loss_tracker = {}
    acc_tracker  = {}
    f1_tracker   = {}

    # ── continual learning trackers ──────────────
    task_accuracy_matrix = {t: {} for t in range(tot_time_steps)}
    bwt_tracker          = {}

    ACCURACY_THRESHOLD = 0.95

    # ════════════════════════════════════════════════════════
    # MAIN LOOP
    # ════════════════════════════════════════════════════════
    for t in range(tot_time_steps):
        print(f"\n{'='*40}")
        print(f"--- Time Step {t+1} / {tot_time_steps} ---")
        print(f"{'='*40}")

        loss_tracker[t] = []
        acc_tracker[t]  = []
        f1_tracker[t]   = []

        best_acc_so_far = -1.0
        active_clients  = client_list

        # ── dynamic client selection ─────────────
        if t > 0:
            print(
                f"\nEvaluating previous best model to "
                f"select clients for Time Step {t+1}..."
            )
            prev_model_path = os.path.join(
                models_dir, f"best_global_model_ts_{t}.pth"
            )
            model.load_state_dict(
                torch.load(prev_model_path, map_location=device)
            )

            active_clients = []
            for c in client_list:
                _, initial_acc, _ = c.evaluate_global_model(
                    model.state_dict(), time_step=t
                )
                print(
                    f"  Client {c.client_id} initial acc "
                    f"on new data: {initial_acc:.4f}"
                )
                if initial_acc < ACCURACY_THRESHOLD:
                    active_clients.append(c)

            print(
                f"\n-> Selected {len(active_clients)} clients "
                f"with accuracy < {ACCURACY_THRESHOLD*100}% "
                f"for training."
            )

            if len(active_clients) == 0:
                print(
                    "All clients performing well. "
                    "Skipping training for this time step."
                )
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        models_dir,
                        f"best_global_model_ts_{t+1}.pth"
                    )
                )

        # ── global iteration loop ─────────────────
        if len(active_clients) > 0:
            for iter in range(args.global_iters):
                local_model_state_dicts = []
                total_loss = 0.0
                total_acc  = 0.0
                total_f1   = 0.0

                print(
                    f"\n--- Global Iteration "
                    f"{iter+1} / {args.global_iters} ---"
                )

                # ── local training ────────────────
                for c in active_clients:
                    model_state = c.train(
                        model.state_dict(), time_step=t
                    )
                    local_model_state_dicts.append(model_state)

                # ── FedAvg aggregation ────────────
                global_state_dict = model.state_dict()
                for key in global_state_dict.keys():
                    global_state_dict[key] = (
                        1 / len(local_model_state_dicts)
                    ) * sum(
                        local_state[key]
                        for local_state in local_model_state_dicts
                    )
                model.load_state_dict(global_state_dict)

                # ── evaluate on ALL clients ───────
                print(
                    f"\n--- Evaluating Global Model on "
                    f"ALL clients at Time Step {t+1} ---"
                )
                for c in client_list:
                    eval_loss, acc, f1 = c.evaluate_global_model(
                        model.state_dict(), time_step=t
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
                    f"Average Accuracy across ALL clients: "
                    f"{avg_acc:.4f}"
                )

                # ── save best model ───────────────
                if avg_acc > best_acc_so_far:
                    best_acc_so_far = avg_acc
                    model_filename  = f"best_global_model_ts_{t+1}.pth"
                    model_filepath  = os.path.join(
                        models_dir, model_filename
                    )
                    torch.save(model.state_dict(), model_filepath)
                    print(
                        f"  --> [Saved] New best model for "
                        f"Time Step {t+1}: {model_filename}"
                    )

                # sync local models with global
                for c in client_list:
                    c.local_model.load_state_dict(model.state_dict())

        # ── backward transfer (BWT) evaluation ───
        print(
            f"\n--- Evaluating Forgetting (BWT) "
            f"at end of Time Step {t+1} ---"
        )

        current_best_path = os.path.join(
            models_dir, f"best_global_model_ts_{t+1}.pth"
        )
        model.load_state_dict(
            torch.load(current_best_path, map_location=device)
        )

        for past_t in range(t + 1):
            past_total_acc = 0.0
            for c in client_list:
                _, acc, _ = c.evaluate_global_model(
                    model.state_dict(), time_step=past_t
                )
                past_total_acc += acc

            avg_past_acc = past_total_acc / len(client_list)
            task_accuracy_matrix[t][past_t] = avg_past_acc
            print(
                f"  Performance on Time Step {past_t+1} "
                f"data: {avg_past_acc:.4f}"
            )

        if t == 0:
            bwt_tracker[t] = 0.0
        else:
            bwt_sum = 0.0
            for past_t in range(t):
                current_acc  = task_accuracy_matrix[t][past_t]
                original_acc = task_accuracy_matrix[past_t][past_t]
                bwt_sum += (current_acc - original_acc)

            bwt_tracker[t] = bwt_sum / t
            print(
                f"\n--> BWT for Time Step {t+1}: "
                f"{bwt_tracker[t]:.4f}"
            )
            if bwt_tracker[t] < 0:
                print(
                    f"    (Forgot "
                    f"{abs(bwt_tracker[t])*100:.2f}% "
                    f"of past knowledge on average)"
                )

    # ── save results ─────────────────────────────
    print("\n--- Experience Replay Training Complete ---")

    results_to_save = {
        "loss"                : loss_tracker,
        "accuracy"            : acc_tracker,
        "f1_score"            : f1_tracker,
        "task_accuracy_matrix": task_accuracy_matrix,
        "bwt"                 : bwt_tracker
    }

    # collect buffer quality (KS test results) from all clients
    buffer_quality = {}
    for c in client_list:
        for domain_key, eval_result in c.buffer_eval_results.items():
            buffer_quality[f'client_{c.client_id}_{domain_key}'] = \
                eval_result['quality']

    results_to_save['buffer_quality'] = buffer_quality

    results_folder = os.path.join(project_root, 'results')
    save_results_as_json(
        'replay_metrics.json',
        results_to_save,
        project_root,
        results_folder
    )
