import json
import os

import torch
from client import Client
from utils import save_results_as_json


def server(args, model, device, domains_path, client_distributions, max_client_participants, tot_time_steps, project_root):
    """
    Executes the Federated Averaging (FedAvg) server logic with dynamic client selection
    and Backward Transfer (BWT) evaluation.
    """
    print("\n--- Starting Federated Averaging (FedAvg) ---")
    
    models_dir = os.path.join(project_root, 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    
    client_list = []
    
    for i in range(0, max_client_participants):
        c = Client(client_id=i, args=args, domain_path=domains_path, assigned_domains=client_distributions[i], device=device, model=model)
        client_list.append(c)
        
    print(f"\nInitialized {len(client_list)} clients with their respective domain data.")
    
    # Trackers for metrics
    loss_tracker = {}    
    acc_tracker = {}
    f1_tracker = {}
    
    # Continual Learning Trackers
    task_accuracy_matrix = {t: {} for t in range(tot_time_steps)} # R_{t, i}
    bwt_tracker = {} # BWT_t
    
    ACCURACY_THRESHOLD = 0.99 

    for t in range(tot_time_steps):
        print(f"\n{'='*40}")
        print(f"--- Time Step {t+1} / {tot_time_steps} ---")
        print(f"{'='*40}")
        loss_tracker[t] = []
        acc_tracker[t] = []
        f1_tracker[t] = []
        
        best_acc_so_far = -1.0
        active_clients = client_list

        # --- Dynamic Client Selection ---
        if t > 0:
            print(f"\nEvaluating previous best model to select clients for Time Step {t+1}...")
            prev_model_path = os.path.join(models_dir, f"fedavg/best_global_model_ts_{t}.pth")
            model.load_state_dict(torch.load(prev_model_path, map_location=device))
            
            active_clients = []
            for c in client_list:
                _, initial_acc, _ = c.evaluate_global_model(model.state_dict(), time_step=t)
                print(f"  Client {c.client_id} initial accuracy on new data: {initial_acc:.4f}")
                
                if initial_acc < ACCURACY_THRESHOLD:
                    active_clients.append(c)
            
            print(f"\n-> Selected {len(active_clients)} clients with accuracy < {ACCURACY_THRESHOLD*100}% for training.")
            
            if len(active_clients) == 0:
                print("All clients are performing well on new data. Skipping training for this time step.")
                torch.save(model.state_dict(), os.path.join(models_dir, f"best_global_model_ts_{t+1}.pth"))
                
                # We still need to evaluate BWT even if we skip training
                pass 
            
        # --- Global Iteration Loop ---
        if len(active_clients) > 0:
            for iter in range(args.global_iters):
                local_model_state_dict = []
                total_loss, total_acc, total_f1 = 0.0, 0.0, 0.0
                
                print(f"\n--- Global Iteration {iter+1} / {args.global_iters} ---")

                for c in active_clients:  
                    model_state = c.train(model.state_dict(), time_step=t)
                    local_model_state_dict.append(model_state)
                
                global_state_dict = model.state_dict()
                for key in global_state_dict.keys():
                    global_state_dict[key] = (1/len(local_model_state_dict)) * sum(local_state[key] for local_state in local_model_state_dict)
                
                model.load_state_dict(global_state_dict)
                
                print(f"\n--- Evaluating Global Model on ALL clients at Time Step {t+1} ---")
                for c in client_list:
                    eval_loss, acc, f1 = c.evaluate_global_model(model.state_dict(), time_step=t)
                    total_loss += eval_loss
                    total_acc += acc
                    total_f1 += f1
                    
                avg_loss = total_loss / len(client_list)
                avg_acc = total_acc / len(client_list)
                avg_f1 = total_f1 / len(client_list)
                
                loss_tracker[t].append(avg_loss)
                acc_tracker[t].append(avg_acc)
                f1_tracker[t].append(avg_f1)
                
                print(f"Average Accuracy across ALL clients: {avg_acc:.4f}")

                if avg_acc > best_acc_so_far:
                    best_acc_so_far = avg_acc
                    model_filename = f"fedavg/best_global_model_ts_{t+1}.pth"
                    model_filepath = os.path.join(models_dir, model_filename)
                    torch.save(model.state_dict(), model_filepath)
                    print(f"  --> [Saved] New best model for Time Step {t+1} saved to {model_filename}")
                
                for c in client_list:
                    c.local_model.load_state_dict(model.state_dict())

        # --- NEW CODE: Backward Transfer (BWT) Evaluation ---
        print(f"\n--- Evaluating Forgetting (BWT) at the end of Time Step {t+1} ---")
        
        # 1. Load the BEST model from the current time step
        current_best_model_path = os.path.join(models_dir, f"fedavg/best_global_model_ts_{t+1}.pth")
        model.load_state_dict(torch.load(current_best_model_path, map_location=device))
        
        # 2. Evaluate it on all past and current time steps (0 to t)
        for past_t in range(t + 1):
            past_total_acc = 0.0
            for c in client_list:
                _, acc, _ = c.evaluate_global_model(model.state_dict(), time_step=past_t)
                past_total_acc += acc
            
            avg_past_acc = past_total_acc / len(client_list)
            task_accuracy_matrix[t][past_t] = avg_past_acc
            print(f"  Performance on Time Step {past_t+1} data: {avg_past_acc:.4f}")
            
        # 3. Calculate BWT
        if t == 0:
            bwt_tracker[t] = 0.0 # No past tasks to forget yet
        else:
            # BWT = average of (current accuracy on past task - original accuracy on past task)
            bwt_sum = 0.0
            for past_t in range(t):
                current_acc = task_accuracy_matrix[t][past_t]   # R_{t, i}
                original_acc = task_accuracy_matrix[past_t][past_t] # R_{i, i}
                bwt_sum += (current_acc - original_acc)
                
            bwt_tracker[t] = bwt_sum / t
            print(f"\n--> Backward Transfer (BWT) for Time Step {t+1}: {bwt_tracker[t]:.4f}")
            if bwt_tracker[t] < 0:
                print(f" (The model has forgotten {abs(bwt_tracker[t])*100:.2f}% of its past knowledge on average)")
        # ----------------------------------------------------

    print("\n--- Training Complete ---")

    # Include the BWT metrics in your JSON save
    results_to_save = {
        "loss": loss_tracker,
        "accuracy": acc_tracker,
        "f1_score": f1_tracker,
        "task_accuracy_matrix": task_accuracy_matrix,
        "bwt": bwt_tracker
    }
    
    results_folder = os.path.join(project_root, 'results')
    save_results_as_json('fedavg_metrics.json', results_to_save, project_root, results_folder)
