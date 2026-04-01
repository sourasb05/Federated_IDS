# server_TimeVAE.py
#
# FedAvg server for the Base TimeVAE generative-replay experiment.
#
# Structurally identical to server_TVAE.py.
# The ONLY differences:
#   - Uses ClientTimeVAE instead of ClientTVAE
#   - Saves results to 'timevae_metrics.json'
#   - Models saved under 'saved_models_timevae/'
#
# The server never sees, shares, or aggregates generators.
# Generators live entirely on each client — private per client.
# The server only aggregates LSTM classifier weights (FedAvg).

import os
import torch

from client_TimeVAE import ClientTimeVAE
from utils import save_results_as_json


def server_timevae(args, model, device, domains_path,
                   client_distributions, max_client_participants,
                   tot_time_steps, project_root):
    """
    Executes FedAvg server logic with ClientTimeVAE clients.

    Each client privately maintains one frozen TimeVAE generator
    per domain and replays past domains during local training.
    The server performs standard FedAvg aggregation on classifier weights.
    """
    print("\n--- Starting FedAvg with TimeVAE Generative Replay ---")

    models_dir = os.path.join(project_root, 'saved_models_timevae')
    os.makedirs(models_dir, exist_ok=True)

    # ── initialise clients ───────────────────────────────────────
    client_list = []
    for i in range(max_client_participants):
        c = ClientTimeVAE(
            client_id        = i,
            args             = args,
            domain_path      = domains_path,
            assigned_domains = client_distributions[i],
            device           = device,
            model            = model,
        )
        client_list.append(c)

    print(f"\nInitialized {len(client_list)} ClientTimeVAE clients.")

    # ── metric trackers ──────────────────────────────────────────
    loss_tracker = {}
    acc_tracker  = {}
    f1_tracker   = {}

    # ── continual learning trackers ─────────────────────────────
    task_accuracy_matrix = {t: {} for t in range(tot_time_steps)}
    bwt_tracker          = {}

    ACCURACY_THRESHOLD = 0.95

    # ════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ════════════════════════════════════════════════════════════
    for t in range(tot_time_steps):
        print(f"\n{'='*40}")
        print(f"--- Time Step {t+1} / {tot_time_steps} ---")
        print(f"{'='*40}")

        loss_tracker[t] = []
        acc_tracker[t]  = []
        f1_tracker[t]   = []

        best_acc_so_far = -1.0
        active_clients  = client_list

        # ── dynamic client selection ──────────────────────────
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
                f"with accuracy < {ACCURACY_THRESHOLD*100:.0f}% "
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

        # ── global iteration loop ─────────────────────────────
        if len(active_clients) > 0:
            for iteration in range(args.global_iters):
                local_model_state_dicts = []
                total_loss = 0.0
                total_acc  = 0.0
                total_f1   = 0.0

                print(
                    f"\n--- Global Iteration "
                    f"{iteration+1} / {args.global_iters} ---"
                )

                # local training
                # ClientTimeVAE.train() handles:
                #   1. Train TimeVAE generator on current domain (once)
                #   2. Build replay loader from past generators
                #   3. Train LSTM on real + replay batches
                for c in active_clients:
                    model_state = c.train(
                        model.state_dict(), time_step=t
                    )
                    local_model_state_dicts.append(model_state)

                # FedAvg aggregation
                global_state_dict = model.state_dict()
                for key in global_state_dict.keys():
                    global_state_dict[key] = (
                        1.0 / len(local_model_state_dicts)
                    ) * sum(
                        local_state[key]
                        for local_state in local_model_state_dicts
                    )
                model.load_state_dict(global_state_dict)

                # evaluate on ALL clients
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

                # save best model
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
                    c.local_model.load_state_dict(
                        model.state_dict()
                    )

        # ── backward transfer (BWT) evaluation ───────────────
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
                    f"{abs(bwt_tracker[t]) * 100:.2f}% "
                    f"of past knowledge on average)"
                )

    # ── save results ─────────────────────────────────────────────
    print("\n--- TimeVAE Training Complete ---")

    results_to_save = {
        "loss"                : loss_tracker,
        "accuracy"            : acc_tracker,
        "f1_score"            : f1_tracker,
        "task_accuracy_matrix": task_accuracy_matrix,
        "bwt"                 : bwt_tracker,
    }

    # collect generator quality scores from all clients
    generator_quality = {}
    for c in client_list:
        for key, eval_result in c.generator_eval_results.items():
            client_key = f'client_{c.client_id}_{key}'
            generator_quality[client_key] = eval_result['quality']

    results_to_save['generator_quality'] = generator_quality

    results_folder = os.path.join(project_root, 'results')
    save_results_as_json(
        'timevae_metrics.json',
        results_to_save,
        project_root,
        results_folder,
    )
