import torch
import numpy as np
import os
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from config import Config
from tqdm import tqdm
from typing import Dict, List
import math
import pandas as pd
from pcgrad import PCGrad

class Trainer:
    def __init__(self, model, config, data_processor, data_path, train_loader, 
                learning_rate=None, include_base_model_params=True,
                base_model_lr_multiplier=1.0, num_epochs=None, use_early_stopping=True,
                fold_info=None):
        self.model = model
        self.config = config
        self.data_processor = data_processor
        self.data_path = data_path
        self.train_loader = train_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate if learning_rate is not None else self.config.LEARNING_RATE
        self.include_base_model_params = include_base_model_params
        self.base_model_lr_multiplier = base_model_lr_multiplier  # Always 1.0 by default parameter
        self.num_epochs = num_epochs if num_epochs is not None else self.config.NUM_EPOCHS
        self.use_early_stopping = use_early_stopping
        self.fold_info = fold_info  # Add fold information
        self.model.to(self.device)
        if hasattr(self.data_processor, 'model') and self.data_processor.model is not None:
             self.data_processor.model.to(self.device)
        self.pos_weights = self._compute_pos_weights()
        self.log_sigma = torch.nn.Parameter(
            torch.zeros(len(self.config.TARGET_COLUMNS), dtype=torch.float32).to(self.device),
            requires_grad=True
        )

    def _compute_pos_weights(self):
        pos_weights = []
        df = pd.read_csv(self.data_path)
        df_filled = df.fillna(-1)
        y = self.data_processor.preprocess_data(df_filled)  # 获取正确的二维标签

        # 添加维度验证
        assert y.ndim == 2, f"标签应为二维数组，实际维度：{y.shape}"

        for i, target in enumerate(self.config.TARGET_COLUMNS):
            labels = y[:, i]  # 现在可以安全使用二维索引
            positive_count = np.sum(labels == 1)
            negative_count = np.sum(labels == 0)
            weight = negative_count / (positive_count + 1e-6) if positive_count > 0 else 1.0
            pos_weights.append(min(weight, 15.0))
        return torch.tensor(pos_weights, dtype=torch.float32).to(self.device)

    def _apply_label_smoothing(self, targets):
        smoothed_targets = targets.clone()
        smoothed_targets[smoothed_targets == 1] = 1 - self.config.LABEL_SMOOTHING
        smoothed_targets[smoothed_targets == 0] = self.config.LABEL_SMOOTHING
        return smoothed_targets

    def train(self, train_loader, val_loader=None):
        # Create parameter groups with different learning rates
        param_groups = []
        
        # Task-specific parameters (MultiTaskTransformer + log_sigma)
        param_groups.append({
            'params': list(self.model.parameters()) + [self.log_sigma],
            'lr': self.learning_rate,
            'weight_decay': self.config.WEIGHT_DECAY
        })
        if self.include_base_model_params and hasattr(self.data_processor, 'model'):
            base_lr = self.learning_rate * self.base_model_lr_multiplier
            print(f"Including base model parameters with lower LR: {base_lr:.2e}")
            param_groups.append({
                'params': list(self.data_processor.model.parameters()),
                'lr': base_lr,
                'weight_decay': self.config.WEIGHT_DECAY
            })
            print("Base model parameters INCLUDED in optimization with reduced learning rate")
        else:
            print("Base model parameters NOT included in optimization")   
        
        optimizer = torch.optim.AdamW(param_groups)
        
        # Wrap with PCGrad if enabled
        if self.config.USE_PCGRAD:
            optimizer = PCGrad(optimizer, reduction=self.config.PCGRAD_REDUCTION)
            print(f"Using PCGrad optimizer with '{self.config.PCGRAD_REDUCTION}' reduction")
        
        # Print learning rates for each parameter group for verification
        for i, group in enumerate(optimizer.param_groups):
            print(f"Parameter group {i}: Learning rate = {group['lr']:.2e}")
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        min_delta = 0.001
        history = {'train_loss': [], 'val_loss': [], 'val_roc_auc': [], 'val_metrics': []}
        
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            task_losses_raw_epoch = torch.zeros(len(self.config.TARGET_COLUMNS), device=self.device) 
            num_task_samples = torch.zeros(len(self.config.TARGET_COLUMNS), device=self.device)
            num_batches_processed = 0

            # Create progress bar with fold information if available
            fold_str = f"[Fold {self.fold_info}] " if self.fold_info else ""
            progress_bar = tqdm(train_loader, desc=f'{fold_str}Epoch {epoch+1}/{self.num_epochs}')
            
            for batch in progress_bar:
                targets = batch["labels"].to(self.device)
                smiles_list = batch["smiles"]

                # Forward pass
                outputs = self.model(smiles_list)
                
                # Calculate task-specific losses for PCGrad
                per_task_losses = []
                task_losses_values = []
                valid_tasks = 0
                
                for t in range(len(self.config.TARGET_COLUMNS)):
                    mask_t = (targets[:, t] != -1)
                    if mask_t.any():
                        outputs_t = outputs[mask_t, t]
                        targets_t = targets[mask_t, t]
                        
                        # Apply label smoothing
                        smoothed_targets_t = self._apply_label_smoothing(targets_t)
                        
                        # Calculate loss with positive class weights
                        loss_fn_t = torch.nn.BCEWithLogitsLoss(
                            reduction='mean',
                            pos_weight=torch.tensor([self.pos_weights[t]], device=self.device)
                        )
                        
                        # Calculate task loss
                        task_loss = loss_fn_t(outputs_t.unsqueeze(1), smoothed_targets_t.unsqueeze(1))
                        
                        # Store the raw task loss for reporting
                        task_losses_raw_epoch[t] += task_loss.item() * mask_t.sum().item()
                        num_task_samples[t] += mask_t.sum().item()
                        task_losses_values.append(task_loss.item())
                        
                        # Apply uncertainty weighting
                        sigma_t = torch.exp(self.log_sigma[t])
                        weighted_loss_t = (task_loss / (2 * sigma_t ** 2)) + self.log_sigma[t]
                        
                        per_task_losses.append(weighted_loss_t)
                        valid_tasks += 1
                
                # Skip batch if no valid tasks
                if valid_tasks == 0:
                    continue
                
                # Add regularization to each task loss
                sigma_l2_reg = self.config.SIGMA_L2_REG * torch.sum(self.log_sigma ** 2) / len(per_task_losses)
                per_task_losses = [loss + sigma_l2_reg for loss in per_task_losses]
                
                # Calculate combined loss for reporting
                total_weighted_loss = sum(per_task_losses)
                
                # Optimization step with PCGrad if enabled
                optimizer.zero_grad()
                if self.config.USE_PCGRAD:
                    # Add retain_graph=True to ensure backward works correctly with multiple losses
                    optimizer.step(per_task_losses, retain_graph=True)
                else:
                    total_weighted_loss.backward()
                    # Gradient clipping
                    if self.include_base_model_params and hasattr(self.data_processor, 'model'):
                        torch.nn.utils.clip_grad_norm_(
                            list(self.model.parameters()) + 
                            [self.log_sigma] + 
                            list(self.data_processor.model.parameters()), 
                            1.0
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            list(self.model.parameters()) + 
                            [self.log_sigma], 
                            1.0
                        )
                    optimizer.step()
                
                running_loss += total_weighted_loss.item()
                num_batches_processed += 1
                
                progress_bar.set_postfix({'loss': running_loss / num_batches_processed if num_batches_processed > 0 else 0,
                                         'tasks': f"{valid_tasks}"})
            
            current_lr = optimizer.param_groups[0]['lr']
            avg_train_loss = running_loss / num_batches_processed if num_batches_processed > 0 else 0
            history['train_loss'].append(avg_train_loss)
            fold_prefix = f"[Fold {self.fold_info}] " if self.fold_info else ""
            print(f"\n{fold_prefix}Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, LR={current_lr:.6f}")

            # Report average RAW task losses for the epoch
            print(f"{fold_prefix}Avg Raw Task Losses (Epoch):")
            for t, target in enumerate(self.config.TARGET_COLUMNS):
                if num_task_samples[t] > 0:
                    avg_raw_loss = task_losses_raw_epoch[t].item() / num_task_samples[t].item()
                else:
                    avg_raw_loss = 0.0
                print(f"  {target}: {avg_raw_loss:.4f}")

            print(f"{fold_prefix}Uncertainty Weights (1/(2*sigma^2)):")
            for t, target in enumerate(self.config.TARGET_COLUMNS):
                sigma_t = torch.exp(self.log_sigma[t])
                weight_t = 1 / (2 * sigma_t ** 2)
                print(f"  {target}: {weight_t.item():.4f}")

            # --- Validation ---
            if val_loader is not None:
                val_loss, val_metrics = self.evaluate_with_loss(val_loader)
                history['val_loss'].append(val_loss)
                history['val_metrics'].append(val_metrics)
                
                # Calculate average ROC AUC from metrics safely
                valid_roc_aucs = [val_metrics[target]['roc_auc'] 
                                  for target in val_metrics.keys() 
                                  if not math.isnan(val_metrics[target]['roc_auc'])]
                avg_roc_auc = np.mean(valid_roc_aucs) if valid_roc_aucs else 0.0
                history['val_roc_auc'].append(avg_roc_auc)
                
                print(f"{fold_prefix}Epoch {epoch+1}: Val Loss={val_loss:.4f}, Avg ROC AUC={avg_roc_auc:.4f}")
                for target in self.config.TARGET_COLUMNS:
                    if target in val_metrics and not math.isnan(val_metrics[target]['roc_auc']):
                        print(f"  {target}: ROC AUC={val_metrics[target]['roc_auc']:.4f}")

                # Only apply early stopping if configured
                if self.use_early_stopping:
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        epochs_without_improvement = 0
                        torch.save(self.model.state_dict(), 'best_model.pt')
                        print(f"{fold_prefix}Validation loss improved to {best_val_loss:.4f}. Saved best model.")
                    else:
                        epochs_without_improvement += 1
                        print(f"{fold_prefix}Validation loss did not improve. Epochs without improvement: {epochs_without_improvement}/{self.config.PATIENCE}")
                        if epochs_without_improvement >= self.config.PATIENCE:
                            print(f'{fold_prefix}Early stopping triggered after epoch {epoch+1}')
                            break
                else:
                    # Just save the model but don't do early stopping
                    if val_loss < best_val_loss:
                        torch.save(self.model.state_dict(), 'best_model.pt')
                        print(f"{fold_prefix}Validation loss improved to {val_loss:.4f}. Saved model.")
            else:
                # If no validation loader provided, just record NaN values
                history['val_loss'].append(float('nan'))
                history['val_metrics'].append({})
                history['val_roc_auc'].append(float('nan'))
                print("No validation loader provided. Skipping validation.")

        self.plot_training_curves(history)
        if val_loader and os.path.exists('best_model.pt'):
            print("Loading best model weights found during training.")
            self.model.load_state_dict(torch.load('best_model.pt'))
        return history

    def evaluate_with_loss(self, data_loader):
        self.model.eval()
        y_true = {target: [] for target in self.config.TARGET_COLUMNS}
        y_pred = {target: [] for target in self.config.TARGET_COLUMNS}
        y_score = {target: [] for target in self.config.TARGET_COLUMNS}
        val_total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                targets = batch["labels"].to(self.device)
                smiles_list = batch["smiles"]

                outputs = self.model(smiles_list)
                batch_weighted_val_loss = torch.tensor(0.0).to(self.device)
                
                # Track predictions for metrics (unchanged)
                for t, target in enumerate(self.config.TARGET_COLUMNS):
                    mask_t = (targets[:, t] != -1)
                    if mask_t.any():
                        outputs_t = outputs[mask_t, t]
                        targets_true = targets[mask_t, t]

                        # Store predictions for metrics (unchanged)
                        targets_true_list = targets_true.cpu().tolist()
                        y_true[target].extend(targets_true_list)
                        logits = outputs_t.cpu()
                        predicted_scores = torch.sigmoid(logits).tolist()
                        predicted_labels = [1 if score > 0.5 else 0 for score in predicted_scores]
                        y_pred[target].extend(predicted_labels)
                        y_score[target].extend(predicted_scores)
                        
                        # Calculate weighted loss similar to training
                        smoothed_targets_t = self._apply_label_smoothing(targets_true)
                        loss_fn_t = torch.nn.BCEWithLogitsLoss(
                            reduction='mean',
                            pos_weight=torch.tensor([self.pos_weights[t]], device=self.device)
                        )
                        L_t = loss_fn_t(outputs_t.unsqueeze(1), smoothed_targets_t.unsqueeze(1))
                        sigma_t = torch.exp(self.log_sigma[t])
                        weighted_loss_t = (L_t / (2 * sigma_t ** 2)) + self.log_sigma[t]
                        batch_weighted_val_loss += weighted_loss_t

                # Simply accumulate the total weighted loss like in training
                if batch_weighted_val_loss > 0:
                    val_total_loss += batch_weighted_val_loss.item()
                    total_batches += 1

        # Calculate average validation loss the same way as training
        avg_val_loss = val_total_loss / total_batches if total_batches > 0 else float('inf')

        # Rest of the function for calculating metrics remains unchanged
        # --- Calculate metrics ---
        metrics = {}
        for target in self.config.TARGET_COLUMNS:
            if len(y_true[target]) > 0:
                try:
                    # Ensure there are both classes present for ROC AUC
                    roc_auc_val = float('nan')
                    if len(set(y_true[target])) > 1:
                         roc_auc_val = roc_auc_score(y_true[target], y_score[target])

                    metrics[target] = {
                        'roc_auc': roc_auc_val
                    }
                except Exception as e:
                    print(f"Error computing metrics for {target}: {e}")
                    metrics[target] = {k: float('nan') for k in ['roc_auc']}
            else:
                metrics[target] = {k: float('nan') for k in ['roc_auc']}

        return avg_val_loss, metrics

    def plot_training_curves(self, history, suffix=''):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['train_loss'], label='Training Loss')
        if 'val_loss' in history and len(history['val_loss']) > 0:
            val_epochs = range(1, len(history['val_loss']) + 1)
            plt.plot(val_epochs, history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        if 'val_roc_auc' in history and len(history['val_roc_auc']) > 0:
            roc_epochs = range(1, len(history['val_roc_auc']) + 1)
            plt.plot(roc_epochs, history['val_roc_auc'], label='Validation ROC AUC', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('ROC AUC')
        plt.title('Validation ROC AUC')
        plt.legend()
        plt.tight_layout()
        filename = f'training_curves{suffix}.png'
        plt.savefig(filename)
        print(f"Saved training curves to {filename}")
        plt.close()
        
        # Plot uncertainty weights
        plt.figure(figsize=(10, 6))
        task_weights = []
        task_names = []
        for t, target in enumerate(self.config.TARGET_COLUMNS):
            sigma_t = torch.exp(self.log_sigma[t]).item()
            weight_t = 1 / (2 * sigma_t ** 2)
            task_weights.append(weight_t)
            task_names.append(target)
        
        plt.bar(range(len(task_weights)), task_weights)
        plt.xticks(range(len(task_weights)), task_names, rotation=45, ha='right')
        plt.ylabel('Weight (1/2σ²)')
        plt.title('Learned Task Weights')
        plt.tight_layout()
        filename = f'task_weights{suffix}.png'
        plt.savefig(filename)
        print(f"Saved task weights to {filename}")
        plt.close()

    def plot_metrics(self, metrics, title, suffix=''):
        targets = list(metrics.keys())
        roc_aucs = [metrics[target]['roc_auc'] for target in targets]
        valid_roc_aucs = [roc for roc in roc_aucs if not math.isnan(roc)]
        valid_roc_indices = [i for i, roc in enumerate(roc_aucs) if not math.isnan(roc)]
        avg_roc_auc = np.mean(valid_roc_aucs) if valid_roc_aucs else 0.0
        print(f"Average ROC AUC Across All Tasks: {avg_roc_auc:.4f}")
        x = np.arange(len(targets))
        width = 0.15
        fig, ax = plt.subplots(figsize=(15, 8))
        if valid_roc_aucs:
            ax.bar([x[i] + 2 * width for i in valid_roc_indices], valid_roc_aucs, width, label='ROC AUC')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 1.1)
        fig.tight_layout()
        filename = f'metrics_plot{suffix}.png'
        plt.savefig(filename, bbox_inches='tight')
        print(f"Saved metrics plot to {filename}")
        plt.close()