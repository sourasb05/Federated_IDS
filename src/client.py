import torch
import torch.nn as nn
import torch.optim as optim
import copy
import utils
from sklearn.metrics import accuracy_score, f1_score

class Client:
    def __init__(self, client_id, args, domain_path, assigned_domains, device, model):
        self.client_id = client_id
        self.args = args
        self.domain_path = domain_path
        self.assigned_domains = assigned_domains
        self.device = device
        self.local_model = copy.deepcopy(model).to(device)
        self.eval_model = copy.deepcopy(model).to(device) 
        # Local metrics storage
        self.local_loss_history = []

        self.train_domains_loader = {}
        self.test_domains_loader = {}
        domains = utils.create_domains(domain_path, assigned_domains)
        for key, files in domains.items():
            self.train_domains_loader[key], self.test_domains_loader[key] = utils.load_data(self.domain_path, key, files, window_size=self.args.window_size, step_size=self.args.step_size, batch_size=self.args.batch_size, n_raw_features=getattr(self.args, 'n_raw_features', None))
            
        # print(f"client {self.client_id} Loaded data for {len(self.train_domains_loader)} domains: {list(self.train_domains_loader.keys())}")
    
        self.domain_keys = list(self.train_domains_loader.keys())
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
    def train(self, global_model_state, time_step):
        """
        Performs local training using the global model's weights.
        Returns the updated state_dict.
        """
        # 1. Initialize local model with current global weights
       
        
        self.local_model.load_state_dict(copy.deepcopy(global_model_state))
        self.local_model.train()

        # 2. Setup Optimizer and Criterion
        

        # print(f"  [Client {self.client_id}] Starting local training...")
        # print(f"  [Client {self.client_id}] Training on domains: {list(self.train_domains_loader.keys())}")
        # 3. Training Loop
        # print(f" [Client {self.client_id}] Time Step {time_step+1}/{len(self.domain_keys)} - Training on domain: {self.domain_keys[time_step]}")
        for epoch in range(self.args.local_epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_domains_loader[self.domain_keys[time_step]]):
                # Ensure data is (batch, seq_len, features)
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output, _ = self.local_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                # Optional: Gradient clipping for LSTMs to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=5.0)
                
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_domains_loader[self.domain_keys[time_step]])
            self.local_loss_history.append(avg_loss)
            # print(f"  [Client {self.client_id}] Epoch {epoch+1}/{self.args.local_epochs}, Loss: {avg_loss:.4f}")
            self.evaluate_local_model(self.local_model.state_dict(), time_step=time_step)

           
            
        # 4. Return the weight delta or new weights
        return self.local_model.state_dict()

    def evaluate_local_model(self, model_state, time_step):
        """
        Optional: Evaluate the local model on this client's local test data.
        """
        from models import LSTMClassifier
        
        self.eval_model.load_state_dict(model_state)
        self.eval_model.eval()
        total_loss = 0.0
        for data, target in self.test_domains_loader[self.domain_keys[time_step]]:  # Just evaluate on the first domain for simplicity
            data, target = data.to(self.device), target.to(self.device)
            with torch.no_grad():
                output, _ = self.eval_model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Compute metrics here (e.g., loss, accuracy) and store them if needed
        evaluateion_loss = total_loss / len(self.test_domains_loader[self.domain_keys[time_step]])
        # print(f"  [Client {self.client_id}] Evaluation Loss: {evaluateion_loss:.4f}")
        
        return evaluateion_loss

        # Use your evaluate_model utility here
        # results = evaluate_model.test(eval_model, self.test_loader, self.device)
        # return results
    
    def evaluate_global_model(self, model_state, time_step):
        """
        Evaluate the local model on this client's local test data.
        Calculates Loss, Accuracy, and F1-Score.
        """
        self.eval_model.load_state_dict(model_state)
        self.eval_model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_domains_loader[self.domain_keys[time_step]]: 
                data, target = data.to(self.device), target.to(self.device)
                
                output, _ = self.eval_model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Get the predicted classes (assuming output is raw logits)
                preds = torch.argmax(output, dim=1)
                
                # Store predictions and targets for sklearn metrics
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
        evaluation_loss = total_loss / len(self.test_domains_loader[self.domain_keys[time_step]])
        
        # Calculate Accuracy and F1-Score 
        # (Using average='weighted' to handle potential class imbalances)
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        print(f"  [Client {self.client_id}] domain : {self.domain_keys[time_step]} Eval Loss: {evaluation_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")
        
        return evaluation_loss, accuracy, f1
    
