from hyperopt import hp, fmin, tpe, Trials
import deepchem as dc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import json
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rdkit import Chem
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import copy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def process_smiles_data(data):

    na_count = data.iloc[:, 1:37].isna().sum(axis=1)
    data = data[na_count < 19]

    # Filter valid SMILES
    data['valid_smiles'] = data['smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)
    data = data[data['valid_smiles']].copy().reset_index(drop=True)

    # Remove duplicates based on 'smiles' column
    data = data.drop_duplicates(subset='smiles').reset_index(drop=True)

    # Standardize SMILES
    def standardize_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else None

    data['smiles'] = data['smiles'].apply(standardize_smiles)

    # Drop any rows with invalid SMILES after standardization
    data = data.dropna(subset=['smiles']).reset_index(drop=True)

    return data


def load_tox21_csv(dataset_file, featurizer='ECFP', split='stratified'):
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
             'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
            ]

    data = pd.read_csv(dataset_file)

    # Preprocess the SMILES column
    data = process_smiles_data(data)

    # Save the preprocessed DataFrame temporarily to a new CSV for loader
    preprocessed_file = "preprocessed_data.csv"
    data.to_csv(preprocessed_file, index=False)

    if featurizer == 'ECFP':
        featurizer = dc.feat.CircularFingerprint()
    elif featurizer == 'MACCS':
        featurizer = dc.feat.MACCSKeysFingerprint()

    loader = dc.data.CSVLoader(tasks=tasks, feature_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(preprocessed_file)

    transformers = [dc.trans.BalancingTransformer(dataset=dataset)]
    for transformer in transformers:
        dataset = transformer.transform(dataset)

    splitters = {
        'stratified': dc.splits.RandomStratifiedSplitter(),
        'scaffold': dc.splits.ScaffoldSplitter()
    }
    splitter = splitters[split]
    train, test = splitter.train_test_split(dataset, seed=42)  # Added seed parameter for reproducibility

    # Return
    return tasks, (train, test), transformers


# Get combination of layer size
def create_subarrays(arr):
    result = []
    for num in arr:
        for i in range(1, 4):
            subarray = [num] * i
            result.append(subarray)
    return result


# Define a function to create a list of dropout rates
def create_dropout_list(layer_sizes, dropout_rate):
    return [dropout_rate] * len(layer_sizes)


# Simplified model architecture
class DNNMultiTaskClassifier(nn.Module):
    """Multi-task DNN classifier with shared layers and task-specific heads

    Architecture Overview:
    1. Shared Feature Processing Layers:
       - Common layers that learn shared representations
       - Multiple FC layers with ReLU activation and dropout
       - Captures common patterns across all tasks

    2. Task-Specific Networks:
       - Separate network branches for each task
       - Each branch has its own hidden layers
       - Allows specialization for each task's unique patterns
       - Final output layer for task-specific prediction

    Parameters:
        n_features (int): Number of input features (2048 for ECFP)
        layer_sizes (list): Sizes of shared processing layers
        dropouts (list): Dropout rates for shared layers
        hidden_layer_sizes (list): Sizes of task-specific hidden layers
        hidden_layer_dropouts (list): Dropout rates for task-specific layers
        n_tasks (int): Number of prediction tasks
    """

    def __init__(self, n_features, layer_sizes, dropouts, hidden_layer_sizes, hidden_layer_dropouts, n_tasks):
        super(DNNMultiTaskClassifier, self).__init__()

        # ===== Shared Feature Processing Layers =====
        self.shared_layers = nn.ModuleList()
        prev_size = n_features

        # Build shared layers
        for size, dropout in zip(layer_sizes, dropouts):
            layer_block = nn.Sequential(
                nn.Linear(prev_size, size),
                nn.BatchNorm1d(size),  # Added batch normalization
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.shared_layers.append(layer_block)
            prev_size = size

        # ===== Task-Specific Networks =====
        self.task_networks = nn.ModuleList()
        final_shared_size = prev_size

        # Create separate network for each task
        for _ in range(n_tasks):
            layers = []
            task_size = final_shared_size

            # Hidden layers for this task
            for size in hidden_layer_sizes:
                task_block = nn.Sequential(
                    nn.Linear(task_size, size),
                    nn.BatchNorm1d(size),  # Added batch normalization
                    nn.ReLU(),
                    nn.Dropout(hidden_layer_dropouts[0])
                )
                layers.append(task_block)
                task_size = size

            # Output layer for this task
            layers.append(nn.Linear(task_size, 1))

            # Combine all layers for this task
            self.task_networks.append(nn.Sequential(*layers))

    def forward(self, x):
        # 1. Flatten input if needed
        x = x.view(x.size(0), -1)

        # 2. Shared feature processing
        for shared_layer in self.shared_layers:
            x = shared_layer(x)

        # 3. Task-specific processing
        task_outputs = []
        for task_net in self.task_networks:
            task_output = task_net(x)
            task_outputs.append(task_output.squeeze(-1))  # shape: (batch_size,)

        # 将列表中的多个张量沿列维度拼接
        output = torch.stack(task_outputs, dim=1)  #shape: (batch_size, n_tasks)

        return output

# Add after the DNNMultiTaskClassifier class definition
def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


num_epochs = 30  # Increase the number of epochs for better training

# Function to minimize
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, mode='classification')



def evaluate_model(model, data_loader, device):
    model.eval()
    y_preds = [[] for _ in range(len(tasks))]
    y_trues = [[] for _ in range(len(tasks))]

    with torch.no_grad():
        for inputs, targets, weights in data_loader:
            inputs, targets, weights = inputs.to(device), targets.to(device), weights.to(device)
            outputs = model(inputs)  # Shape: (batch_size, n_tasks)

            for task_idx in range(len(tasks)):
                # 提取当前任务的输出和目标
                task_pred = outputs[:, task_idx]  # Shape: (batch_size,)
                task_target = targets[:, task_idx]  # Shape: (batch_size,)
                task_weight = weights[:, task_idx]

                # Create valid sample masks (filter-1 and NaN)
                valid_mask = (task_weight != 0).cpu().numpy()

                if np.any(valid_mask):
                    # The prediction and true values of the valid sample are extracted
                    valid_pred = task_pred.cpu().numpy()[valid_mask]
                    valid_true = task_target.cpu().numpy()[valid_mask]

                    y_preds[task_idx].append(valid_pred)
                    y_trues[task_idx].append(valid_true)

    task_scores = []
    for task_idx in range(len(tasks)):
        # Merge predicted and true values for all batches
        task_true = np.concatenate(y_trues[task_idx]) if y_trues[task_idx] else np.array([])
        task_pred = np.concatenate(y_preds[task_idx]) if y_preds[task_idx] else np.array([])

        # Calculate ROC-AUC
        if len(np.unique(task_true)) > 1:
            task_scores.append(dc.metrics.roc_auc_score(task_true, task_pred))
        else:
            task_scores.append(float('nan'))

    return task_scores

# Simplified training function
def log_model_architecture(model):
    """Log detailed model architecture information"""
    logger.info("\n=== Model Architecture ===")
    logger.info(f"Total layers: {len(list(model.modules()))}")
    logger.info("\nShared Layers:")
    for i, layer in enumerate(model.shared_layers):
        logger.info(f"Layer {i + 1}: {layer}")

    logger.info("\nTask-Specific Networks:")
    for i, task_net in enumerate(model.task_networks):
        logger.info(f"\nTask {i + 1} ({tasks[i]}):")
        for j, layer in enumerate(task_net):
            logger.info(f"Layer {j + 1}: {layer}")


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


#  Trains the final model on the entire training set using optimal hyperparameters
def train_final_model(best_params, train_dataset, tasks, num_epochs=30):
    """
    Parameters:
    best_params: The best combination of parameters obtained through hyperparameter optimization
    train_dataset: Complete training dataset
    tasks: indicates a list of tasks
    num_epochs: training round
    Back:
    model: The trained final model
    """
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n=== Training Final Model on {device} ===")

    # Preparing DataLoader
    # --------------------------------------------------
    logger.info("Preparing DataLoader...")

    # Use optimized batch sizes
    batch_size = max(int(best_params['batch_size']), 2)

    # Create a complete training set DataLoader
    train_dataset_targets = [torch.Tensor(train_dataset.y[:, i]).unsqueeze(1) for i in range(len(tasks))]
    train_targets = torch.cat(train_dataset_targets, dim=1)
    train_dataset_weights = [torch.Tensor(train_dataset.w[:, i]).unsqueeze(1) for i in range(len(tasks))]
    train_weights = torch.cat(train_dataset_weights, dim=1)

    train_loader = DataLoader(
        TensorDataset(
            torch.Tensor(train_dataset.X),
            train_targets,
            train_weights,
        ),
        batch_size=batch_size,  # Ensure minimum batch size of 2
        shuffle=True,
        drop_last=True  # Drop last batch if incomplete
    )

    model = DNNMultiTaskClassifier(
        n_features=2048,
        layer_sizes=best_params['layer_sizes'],
        dropouts=create_dropout_list(best_params['layer_sizes'], best_params['dropout_rate']),
        hidden_layer_sizes=best_params['hidden_layers'],
        hidden_layer_dropouts=create_dropout_list(best_params['hidden_layers'],
                                                  best_params['hidden_layer_dropout_rate']),
        n_tasks=len(tasks)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=10)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    best_model = None
    best_score = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets, weights in train_loader:
            inputs, targets, weights = inputs.to(device), targets.to(device), weights.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            losses = []

            for task_idx in range(len(tasks)):
                task_out = outputs[:, task_idx]
                task_target = targets[:, task_idx]
                task_weights = weights[:, task_idx]

                # Calculate losses per sample (unweighted)
                task_loss_per_sample = loss_fn(task_out, task_target)

                # The weights are applied and the average loss of a valid sample is calculated
                weighted_task_loss = task_loss_per_sample * task_weights
                valid_samples = (task_weights != 0).float().sum()

                if valid_samples > 0:
                    task_loss = weighted_task_loss.sum() / valid_samples
                else:
                    task_loss = torch.tensor(0.0, device=device)

                losses.append(task_loss)

            loss = sum(losses) / len(losses)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)
        early_stopping(avg_loss)

        if avg_loss < best_score:
            best_score = avg_loss
            best_model = copy.deepcopy(model.state_dict())
            logger.info(f"New best loss: {best_score:.4f}")

        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            # Ensure that the current best model is saved when an early shutdown is triggered
            if best_model is not None:
                model.load_state_dict(best_model)
            break

    if best_model is None:
        logger.warning("No valid model saved, using last epoch weights")
        best_model = model.state_dict()

    model.load_state_dict(best_model)
    return model


def fm(args):

    global global_best_score, global_best_architecture, global_best_hyperparams
    global all_results

    save_dir = os.path.dirname(os.path.abspath(__file__))
    layer_sizes = args['layer_sizes']
    dropouts = create_dropout_list(layer_sizes, args['dropout_rate'])
    hidden_layer_sizes = args['hidden_layers']
    hidden_layer_dropouts = create_dropout_list(hidden_layer_sizes, args['hidden_layer_dropout_rate'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_model_path = os.path.join(save_dir, "global_best_model.pth")

    kf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    #train_dataset, test_dataset
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_dataset.X, train_dataset.y)):
        logger.info(f"\n=== Fold {fold_idx + 1} ===")


        X_train_fold, X_val_fold = train_dataset.X[train_idx], train_dataset.X[val_idx]
        y_train_fold, y_val_fold = train_dataset.y[train_idx], train_dataset.y[val_idx]
        w_train_fold, w_val_fold = train_dataset.w[train_idx], train_dataset.w[val_idx]

        # Prepare DataLoader
        train_dataset_targets = [torch.Tensor(y_train_fold[:, i]).unsqueeze(1) for i in range(len(tasks))]
        train_targets = torch.cat(train_dataset_targets, dim=1)
        val_dataset_targets = [torch.Tensor(y_val_fold[:, i]).unsqueeze(1) for i in range(len(tasks))]
        val_targets = torch.cat(val_dataset_targets, dim=1)
        train_dataset_weights = [torch.Tensor(w_train_fold[:, i]).unsqueeze(1) for i in range(len(tasks))]
        train_weights = torch.cat(train_dataset_weights, dim=1)
        val_dataset_weights = [torch.Tensor(w_val_fold[:, i]).unsqueeze(1) for i in range(len(tasks))]
        val_weights = torch.cat(val_dataset_weights, dim=1)

        # Prepare DataLoader with drop_last=True
        train_loader = DataLoader(
            TensorDataset(
                torch.Tensor(X_train_fold),
                train_targets,
                train_weights
            ),
            batch_size=max(int(args['batch_size']), 2),  # Ensure minimum batch size of 2
            shuffle=True,
            drop_last=True  # Drop last batch if incomplete
        )

        val_loader = DataLoader(
            TensorDataset(
                torch.Tensor(X_val_fold),
                val_targets,
                val_weights
            ),
            batch_size=max(int(args['batch_size']), 2),  # Ensure minimum batch size of 2
            shuffle=False,
            drop_last=False # Keep all samples for testing
        )

        # Create model
        model = DNNMultiTaskClassifier(n_features=2048, layer_sizes=layer_sizes, dropouts=dropouts,
                                       hidden_layer_sizes=hidden_layer_sizes,
                                       hidden_layer_dropouts=hidden_layer_dropouts,
                                       n_tasks=len(tasks)).to(device)

        # Log model architecture
        log_model_architecture(model)
        total_params = count_parameters(model)
        logger.info(f"\nTotal trainable parameters: {total_params:,}")
        logger.info(f"Device being used: {device}")

        optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        early_stopping = EarlyStopping(patience=10, verbose=True)
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        #train_losses = []
        test_roc_auc_scores = []
        best_score = float('-inf')

        logger.info(f"Training with hyperparameters: {args}")
        logger.info(f"Starting training with {num_epochs} epochs")


        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            task_losses = [0] * len(tasks)
            batch_count = 0

            for batch_idx, (inputs, targets, weights) in enumerate(train_loader):
                inputs, targets, weights = inputs.to(device), targets.to(device), weights.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                # Calculate individual task losses
                losses = []

                for task_idx in range(len(tasks)):
                    task_out = outputs[:, task_idx]
                    task_target = targets[:, task_idx]
                    task_weights = weights[:, task_idx]

                    # Calculate losses per sample (unweighted)
                    task_loss_per_sample = loss_fn(task_out, task_target)

                    # The weights are applied and the average loss of a valid sample is calculated
                    weighted_task_loss = task_loss_per_sample * task_weights
                    valid_samples = (task_weights != 0).float().sum()

                    if valid_samples > 0:
                        task_loss = weighted_task_loss.sum() / valid_samples
                    else:
                        task_loss = torch.tensor(0.0, device=device)

                    losses.append(task_loss)


                loss = sum(losses) / len(losses)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1

                # Log batch progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{num_epochs} - Batch {batch_idx + 1}/{len(train_loader)} - "
                                f"Batch Loss: {loss.item():.4f} - "
                                f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Calculate average losses
            epoch_loss /= batch_count
            task_losses = [t_loss / batch_count for t_loss in task_losses]

            # Evaluate model
            test_scores = evaluate_model(model, val_loader, device)
            test_roc_auc_scores.append(test_scores)

            # Log detailed epoch results
            logger.info(f"\n=== Epoch {epoch + 1}/{num_epochs} Summary ===")
            logger.info(f"Average Loss: {epoch_loss:.4f}")
            logger.info("\nPer-Task Losses:")
            for task_idx, task_loss in enumerate(task_losses):
                logger.info(f"{tasks[task_idx]}: {task_loss:.4f}")
            logger.info("\nPer-Task ROC-AUC Scores:")
            for task_idx, score in enumerate(test_scores):
                logger.info(f"{tasks[task_idx]}: {score:.4f}")
            logger.info(f"Average ROC-AUC: {sum(test_scores) / len(test_scores):.4f}")

            # Calculate validation loss for scheduling
            model.eval()
            val_loss = 0
            total_valid_tasks = 0

            with torch.no_grad():
                for inputs, targets, weights in val_loader:
                    inputs, targets, weights = inputs.to(device), targets.to(device), weights.to(device)
                    outputs = model(inputs)  # Shape: (batch_size, n_tasks)

                    batch_loss = 0.0
                    valid_tasks_in_batch = 0

                    # 遍历每个任务
                    for task_idx in range(len(tasks)):
                        task_out = outputs[:, task_idx]  # Shape: (batch_size,)
                        task_target = targets[:, task_idx]  # Shape: (batch_size,)
                        task_weight = weights[:, task_idx]

                        mask = (task_weight != 0) & (~torch.isnan(task_target))

                        # Extract valid sample
                        valid_out = task_out[mask]
                        valid_target = task_target[mask]

                        # Calculate the loss of the current task (valid sample only)
                        if len(valid_out) > 0:
                            task_loss = loss_fn(valid_out, valid_target).mean()
                            batch_loss += task_loss.item()
                            valid_tasks_in_batch += 1

                    # Add up batch losses and number of active tasks
                    if valid_tasks_in_batch > 0:
                        val_loss += batch_loss / valid_tasks_in_batch
                        total_valid_tasks += 1

            # Final validation losses are averaged by valid lot
            if total_valid_tasks > 0:
                val_loss /= total_valid_tasks
            else:
                val_loss = float('inf')

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break

            # Save best score
            avg_test_score = sum(test_scores) / len(test_scores)
            if avg_test_score > best_score:
                best_score = avg_test_score

        # Record the best score for the fold
        fold_scores.append(best_score)

    # Correctly return 50% off average score
    avg_score = sum(fold_scores) / len(fold_scores)
    logger.info(f"\nTrial Average Score: {avg_score:.4f}")

    # 更新全局最佳
    if avg_score > global_best_score:
        global_best_score = avg_score
        global_best_architecture = {
            'n_features': 2048,
            'layer_sizes': layer_sizes,
            'dropouts': dropouts,
            'hidden_layers': hidden_layer_sizes,
            'hidden_layer_dropouts': hidden_layer_dropouts,
            'batch_size': args['batch_size'],
            'learning_rate': args['learning_rate']
        }
        global_best_hyperparams = args.copy()

        # Save the best global configuration
        torch.save(model.state_dict(), os.path.join(save_dir, "global_best_model.pth"))
        with open(os.path.join(save_dir, "global_best_architecture.json"), "w") as f:
            json.dump(global_best_architecture, f)
        logger.info("New Global Best Model Saved!")

    all_results.append({
        "params": args,
        "fold_scores": fold_scores,
        "score": avg_score
    })

    return -1 * avg_score


if __name__ == "__main__":
    # Load the local csv data file
    data = r'D:\SSH\MTL-DNN\tox21.csv'
    tasks, datasets, transformers = load_tox21_csv(data)
    train_dataset, test_dataset = datasets

    # Print dataset information
    logger.info("\n=== Dataset Information ===")
    logger.info(f"Number of tasks: {len(tasks)}")
    logger.info(f"Tasks: {tasks}")

    logger.info("\n=== Training Dataset ===")
    logger.info(f"Training set shape - X: {train_dataset.X.shape}, y: {train_dataset.y.shape}")
    logger.info("\nFirst 5 samples of training data:")
    for i in range(min(5, len(train_dataset.X))):
        logger.info(f"\nSample {i + 1}:")
        logger.info(f"Features (first 10): {train_dataset.X[i][:10]}...")
        logger.info(f"Labels: {dict(zip(tasks, train_dataset.y[i]))}")

    logger.info("\nLast 5 samples of training data:")
    for i in range(max(0, len(train_dataset.X) - 5), len(train_dataset.X)):
        logger.info(f"\nSample {i + 1}:")
        logger.info(f"Features (first 10): {train_dataset.X[i][:10]}...")
        logger.info(f"Labels: {dict(zip(tasks, train_dataset.y[i]))}")

    logger.info("\n=== Test Dataset ===")
    logger.info(f"Test set shape - X: {test_dataset.X.shape}, y: {test_dataset.y.shape}")
    logger.info("\nFirst 5 samples of test data:")
    for i in range(min(5, len(test_dataset.X))):
        logger.info(f"\nSample {i + 1}:")
        logger.info(f"Features (first 10): {test_dataset.X[i][:10]}...")
        logger.info(f"Labels: {dict(zip(tasks, test_dataset.y[i]))}")

    logger.info("\nLast 5 samples of test data:")
    for i in range(max(0, len(test_dataset.X) - 5), len(test_dataset.X)):
        logger.info(f"\nSample {i + 1}:")
        logger.info(f"Features (first 10): {test_dataset.X[i][:10]}...")
        logger.info(f"Labels: {dict(zip(tasks, test_dataset.y[i]))}")

    # Print class distribution for each task
    logger.info("\n=== Class Distribution ===")
    for i, task in enumerate(tasks):
        train_pos = (train_dataset.y[:, i] == 1).sum()
        train_neg = (train_dataset.y[:, i] == 0).sum()
        test_pos = (test_dataset.y[:, i] == 1).sum()
        test_neg = (test_dataset.y[:, i] == 0).sum()

        logger.info(f"\nTask: {task}")
        logger.info(
            f"Training - Positive: {train_pos}, Negative: {train_neg}, Ratio: {train_pos / (train_pos + train_neg):.3f}")
        logger.info(f"Test - Positive: {test_pos}, Negative: {test_neg}, Ratio: {test_pos / (test_pos + test_neg):.3f}")

    # Add user prompt
    logger.info("\n=== Verification Complete ===")
    logger.info("Please verify the dataset information above.")
    input("Press Enter to start training...")
    logger.info("\nStarting hyperparameter optimization...")

    global_best_score = float('-inf')
    global_best_architecture = None
    global_best_hyperparams = None
    all_results = []

    # Define the grid search parameter space

    arr = [2048, 1024]
    layer_size_comb = create_subarrays(arr)
    logger.info(f"Layer size combinations: {layer_size_comb}")

    search_space = {
        'layer_sizes': hp.choice('layer_sizes', layer_size_comb),
        'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),  # Adjusted range
        'learning_rate': hp.loguniform('learning_rate', -4, -2),  # Adjusted range
        'batch_size': hp.choice('batch_size', [32, 64, 128]),  # Removed smaller batch sizes
        'hidden_layers': hp.choice('hidden_layers', [
            [1024, 512, 256],
            [2048, 1024, 512],
            [512, 256, 128]
        ]),
        'hidden_layer_dropout_rate': hp.uniform('hidden_layer_dropout_rate', 0.1, 0.5)  # Adjusted range
    }

    trials = Trials()
    best = fmin(fm, space=search_space, algo=tpe.suggest, max_evals=20, trials=trials)
    logger.info(f"Best hyperparameters found: {best}")


    with open('grid_search_results.json', 'w') as f:
        json.dump({'all_trials': all_results}, f)

    # Load the best hyperparameters
    save_dir = os.path.dirname(os.path.abspath(__file__))
    global_model_path = os.path.join(save_dir, "global_best_model.pth")

    if os.path.exists(global_model_path):
        # 加载架构
        with open(os.path.join(save_dir, "global_best_architecture.json"), "r") as f:
            best_architecture = json.load(f)

    best_layer_sizes = best_architecture['layer_sizes']
    best_dropout_rate = best_architecture['dropouts'][0]
    best_hidden_layers = best_architecture['hidden_layers']
    best_hidden_layer_dropout_rate = best_architecture['hidden_layer_dropouts'][0]
    best_batch_size = best_architecture['batch_size']
    best_learning_rate = best_architecture['learning_rate']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = DNNMultiTaskClassifier(
        n_features=2048,
        layer_sizes=best_layer_sizes,
        dropouts=create_dropout_list(best_layer_sizes, best_dropout_rate),
        hidden_layer_sizes=best_hidden_layers,
        hidden_layer_dropouts=create_dropout_list(best_hidden_layers, best_hidden_layer_dropout_rate),
        n_tasks=len(tasks)
    ).to(device)

    # Add parameter counting here
    total_params = count_parameters(best_model)
    logger.info(f"Best model trainable parameters: {total_params:,}")
    # Add explicit print statement
    print(f"\nTotal trainable parameters in best model: {total_params:,}")

    # Save parameters count to file
    params_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_parameters.txt")
    with open(params_file, 'w') as f:
        f.write(f"Total trainable parameters: {total_params:,}\n")
        f.write(f"Best layer sizes: {best_layer_sizes}\n")
        f.write(f"Best dropout rate: {best_dropout_rate:.4f}\n")
        f.write(f"Best hidden layers: {best_hidden_layers}\n")
        f.write(f"Best hidden layer dropout rate: {best_hidden_layer_dropout_rate:.4f}\n")
        f.write(f"Best batch size: {best_batch_size}\n")
        f.write(f"Best learning rate: {best_learning_rate:.6f}\n")
    logger.info(f"Model parameters and hyperparameters saved to {params_file}")

    if global_best_hyperparams:
        final_model = train_final_model(global_best_hyperparams, train_dataset, tasks)
        torch.save(final_model.state_dict(), "final_model.pth")

    # Ensure the model architecture matches the saved state
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_model.pth")
    if os.path.exists(model_path):
        best_model.load_state_dict(torch.load(model_path))
        best_model.eval()

        test_dataset_targets = [torch.Tensor(test_dataset.y[:, i]).unsqueeze(1) for i in range(len(tasks))]
        test_targets = torch.cat(test_dataset_targets, dim=1)
        test_dataset_weights = [torch.Tensor(test_dataset.w[:, i]).unsqueeze(1) for i in range(len(tasks))]
        test_weights = torch.cat(test_dataset_weights, dim=1)

        test_loader = DataLoader(
            TensorDataset(torch.Tensor(test_dataset.X), test_targets, test_weights),
            batch_size=best_batch_size,
            shuffle=False
        )


        test_scores = evaluate_model(best_model, test_loader, device)
        logger.info(f"ROC-AUC score for the best hyperparameters: {test_scores}")

        # 保存参数信息
        total_params = count_parameters(best_model)
        with open(os.path.join(save_dir, "final_model_info.txt"), "w") as f:
            f.write(f"Global Best Validation Score: {global_best_score:.4f}\n")
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Hyperparameters: {json.dumps(global_best_hyperparams, indent=2)}\n")
            f.write("Test Scores:\n")
            for task, score in zip(tasks, test_scores):
                f.write(f"{task}: {score:.4f}\n")
            f.write(f"\nAverage ROC-AUC: {sum(test_scores) / len(test_scores):.4f}")
    else:
        logger.error("No Best model found!")