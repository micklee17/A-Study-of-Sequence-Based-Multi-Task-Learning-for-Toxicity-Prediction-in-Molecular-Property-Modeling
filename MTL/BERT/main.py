import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import os
from data_processor import SmilesFinetuneDataset, DataProcessor
from mtl_collator import MTLBatchCollator
from model import MultiTaskTransformer
from train_eval import Trainer
from config import Config
from sklearn.model_selection import train_test_split, KFold
import math
import pickle
import numpy as np
from pathlib import Path

def ensure_model_files(model_name, local_path):
    """Download model if local path doesn't exist or is missing files, and save to local path"""
    local_path = Path(local_path)
    required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.json", "merges.txt"]
    
    # Check if local directory exists and has all required files
    if local_path.exists() and all((local_path / file).exists() for file in required_files):
        print(f"Using existing model files from {local_path}")
        return local_path
    
    print(f"Model files missing or incomplete. Downloading from {model_name}...")
    
    # Create the directory if it doesn't exist
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Download model and tokenizer from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Save model and tokenizer to local path
    model.save_pretrained(local_path)
    tokenizer.save_pretrained(local_path)
    
    print(f"Model and tokenizer saved to {local_path}")
    return local_path


def analyze_data_distribution(train_val_df, test_df, target_columns):
    """Analyze and compare label distributions between train+val and test sets"""
    print("\n===== DATA DISTRIBUTION ANALYSIS =====")
    
    # Compare tox_group distribution (stratification variable)
    print("\nTox Group Distribution (stratification variable):")
    train_tox_groups = train_val_df['tox_group'].value_counts(normalize=True).sort_index() * 100
    test_tox_groups = test_df['tox_group'].value_counts(normalize=True).sort_index() * 100
    
    for group in sorted(set(train_val_df['tox_group'].unique()) | set(test_df['tox_group'].unique())):
        train_pct = train_tox_groups.get(group, 0)
        test_pct = test_tox_groups.get(group, 0)
        print(f"  Group {group}: Train+Val={train_pct:.1f}%, Test={test_pct:.1f}%, Diff={test_pct-train_pct:.1f}%")
    
    # Analyze each endpoint's distribution
    print("\nEndpoint-Specific Distributions:")
    for col in target_columns:
        train_vals = train_val_df[col].fillna(-1).value_counts(normalize=True) * 100
        test_vals = test_df[col].fillna(-1).value_counts(normalize=True) * 100
        
        print(f"\n{col}:")
        for val in [-1, 0, 1]:  # -1=missing, 0=negative, 1=positive
            train_pct = train_vals.get(val, 0)
            test_pct = test_vals.get(val, 0)
            label_name = "Missing" if val == -1 else ("Negative" if val == 0 else "Positive")
            print(f"  {label_name}: Train+Val={train_pct:.1f}%, Test={test_pct:.1f}%, Diff={test_pct-train_pct:.1f}%")
    
    # Calculate overall missingness patterns
    train_missing = (train_val_df[target_columns].isna().sum() / len(train_val_df) * 100).sort_values(ascending=False)
    test_missing = (test_df[target_columns].isna().sum() / len(test_df) * 100).sort_values(ascending=False)
    
    print("\nMissing Value Analysis (sorted by most missing):")
    for col in train_missing.index:
        train_pct = train_missing[col]
        test_pct = test_missing[col]
        print(f"  {col}: Train+Val={train_pct:.1f}% missing, Test={test_pct:.1f}% missing, Diff={test_pct-train_pct:.1f}%")
    
    print("\n===== END DISTRIBUTION ANALYSIS =====")


def main():
    # Initialize configuration
    config = Config()

    # Define the local path and original model names
    local_model_path = "C:/Users/Ford/Desktop/chemberta_no_unfreezing/MTL-BERT/chemberta"
    model_names = {
        1: "seyonec/ChemBERTa-zinc-base-v1",
        2: "pchanda/pretrained-smiles-pubchem10m",
        3: "HUBioDataLab/SELFormer"
    }
    
    try:
        # Ensure model files exist
        if config.PRETRAIN_CHOICE not in model_names:
            raise ValueError("Invalid pre-trained model selection, please select 1, 2 or 3")
        
        model_name = model_names[config.PRETRAIN_CHOICE]
        local_path = ensure_model_files(model_name, local_model_path)
        
        # Load tokenizer
        print(f"Loading tokenizer from {local_path}")
        config.TOKENIZER = AutoTokenizer.from_pretrained(str(local_path))
        
    except Exception as e:
        print(f"Error initializing model/tokenizer: {e}")
        print("Attempting to continue using Hugging Face model directly...")
        # Fallback to using the HF model directly
        model_name = model_names[config.PRETRAIN_CHOICE]
        config.TOKENIZER = AutoTokenizer.from_pretrained(model_name)

    # Load dataset
    data_path = 'C:/Users/Ford/Desktop/chemberta_no_unfreezing/MTL-BERT/tox21.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} not found!")
    print("Loading data...")
    df = pd.read_csv(data_path)

    # Validate dataset
    if df.empty:
        raise ValueError("Loaded DataFrame is empty!")
    if 'smiles' not in df.columns:
        raise ValueError("DataFrame does not contain 'smiles' column!")
    for target in config.TARGET_COLUMNS:
        if target not in df.columns:
            raise ValueError(f"DataFrame does not contain '{target}' column!")

    # Create aggregated toxicity score for stratification
    df['tox_score'] = df[config.TARGET_COLUMNS].fillna(-1).sum(axis=1)
    df['tox_group'] = pd.cut(df['tox_score'], bins=5, labels=False)
    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, 
                                             stratify=df['tox_group'])
    print(f"Dataset split: Train+Val={len(train_val_df)}, Test={len(test_df)}")

    # Analyze data distribution
    analyze_data_distribution(train_val_df, test_df, config.TARGET_COLUMNS)

    # Initialize data processor
    data_processor = DataProcessor(config)
    data_processor.load_pretrained_model()

    # Preprocess test data
    print("Preprocessing test data...")
    y_test = data_processor.preprocess_data(test_df, is_train=False)
    test_dataset = SmilesFinetuneDataset(
        test_df,
        config.TARGET_COLUMNS,
        tokenizer=data_processor.tokenizer,
        data_processor=data_processor
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=MTLBatchCollator(),
        drop_last=False
    )

    # Set up 5-fold cross-validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_df)):
        print(f"\nStarting Fold {fold + 1}/{n_splits}")

        # Split train and validation sets for this fold
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

        # Preprocess data
        print("Preprocessing training data...")
        y_train = data_processor.preprocess_data(train_df, is_train=True)
        print("Preprocessing validation data...")
        y_val = data_processor.preprocess_data(val_df, is_train=False)

        # Create datasets
        train_dataset = SmilesFinetuneDataset(
            train_df,
            config.TARGET_COLUMNS,
            tokenizer=data_processor.tokenizer,
            data_processor=data_processor
        )
        val_dataset = SmilesFinetuneDataset(
            val_df,
            config.TARGET_COLUMNS,
            tokenizer=data_processor.tokenizer,
            data_processor=data_processor
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=MTLBatchCollator(),
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=MTLBatchCollator(),
            drop_last=False
        )

        # Initialize model
        model = MultiTaskTransformer(config, data_processor)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        model.to(device)
        data_processor.model.to(device)

        # Train model
        print(f"Training model for Fold {fold + 1}...")
        trainer = Trainer(
            model, 
            config, 
            data_processor, 
            data_path,
            train_loader,
            learning_rate=config.LEARNING_RATE,
            include_base_model_params=config.INCLUDE_BASE_MODEL_PARAMS,
            num_epochs=config.NUM_EPOCHS,
            use_early_stopping=True,
            fold_info=f"{fold + 1}/{n_splits}"  # Pass fold information
        )
        history = trainer.train(train_loader, val_loader)

        # Evaluate on validation set
        print(f"Evaluating Fold {fold + 1} on validation set...")
        val_loss, val_metrics = trainer.evaluate_with_loss(val_loader)
        fold_results.append({
            'fold': fold + 1,
            'val_loss': val_loss,
            'val_metrics': val_metrics
        })

        # Plot metrics for this fold with fold suffix
        trainer.plot_metrics(val_metrics, f"Model performance across toxic endpoints - Fold {fold + 1}", f"_fold{fold+1}")
        # Add plotting training curves with fold suffix
        trainer.plot_training_curves(history, f"_fold{fold+1}")

    # Aggregate results across folds
    print("\nCross-Validation Results Summary:")
    avg_val_loss = np.mean([result['val_loss'] for result in fold_results])
    print(f"Average Validation Loss: {avg_val_loss:.4f}")

    # Compute average metrics across folds
    avg_metrics = {}
    fold_roc_aucs = {target: [] for target in config.TARGET_COLUMNS}
    
    print("\nROC-AUC scores across all folds:")
    for fold_result in fold_results:
        fold_num = fold_result['fold']
        print(f"Fold {fold_num}:")
        for target in config.TARGET_COLUMNS:
            roc_auc = fold_result['val_metrics'][target]['roc_auc']
            if not math.isnan(roc_auc):
                fold_roc_aucs[target].append(roc_auc)
                print(f"  {target}: {roc_auc:.4f}")
    
    print("\nAverage ROC-AUC across all folds (more reliable estimate):")
    for target in config.TARGET_COLUMNS:
        valid_scores = fold_roc_aucs[target]
        if valid_scores:
            avg_roc_auc = np.mean(valid_scores)
            std_roc_auc = np.std(valid_scores)
            avg_metrics[target] = {'roc_auc': avg_roc_auc}
            print(f"{target}:")
            print(f"  ROC AUC: {avg_roc_auc:.4f} ± {std_roc_auc:.4f}")
            print(f"  Min: {min(valid_scores):.4f}, Max: {max(valid_scores):.4f}")
        else:
            avg_metrics[target] = {'roc_auc': float('nan')}
            print(f"{target}: No valid ROC AUC scores")

    # Train final model on full train+val data, but keep a small monitoring set ONLY for visualization
    print("\nTraining final model on full train+val data...")
    # Create monitoring set from test data (10% of test data)
    test_monitor, test_final = train_test_split(test_df, test_size=0.9, random_state=42)
    print(f"Final model will train on ALL {len(train_val_df)} training examples")
    print(f"Using {len(test_monitor)} examples from test set for progress monitoring ONLY")

    # Create datasets for final training
    train_final_dataset = SmilesFinetuneDataset(
        train_val_df,  # Use ALL train+val data for training
        config.TARGET_COLUMNS,
        tokenizer=data_processor.tokenizer,
        data_processor=data_processor
    )

    # Create a monitoring dataset from test data (never used in training)
    monitor_val_dataset = SmilesFinetuneDataset(
        test_monitor,  # Use subset of test data for monitoring only
        config.TARGET_COLUMNS,
        tokenizer=data_processor.tokenizer,
        data_processor=data_processor
    )

    # For final evaluation, use the remaining test data
    test_final_dataset = SmilesFinetuneDataset(
        test_final,
        config.TARGET_COLUMNS,
        tokenizer=data_processor.tokenizer,
        data_processor=data_processor
    )

    train_final_loader = DataLoader(
        train_final_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=MTLBatchCollator(),
        drop_last=True
    )
    
    monitor_val_loader = DataLoader(
        monitor_val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=MTLBatchCollator(),
        drop_last=False
    )

    # Train final model
    final_model = MultiTaskTransformer(config, data_processor)
    final_model.to(device)
    data_processor.model.to(device)
    
    final_trainer = Trainer(
        final_model,
        config,
        data_processor,
        data_path,
        train_final_loader,
        learning_rate=config.LEARNING_RATE,
        include_base_model_params=config.INCLUDE_BASE_MODEL_PARAMS,
        num_epochs=config.NUM_EPOCHS,
        use_early_stopping=False,  # No early stopping for final model
        fold_info="Final"  # Indicate this is the final model training
    )
    
    # Use monitor_val_loader ONLY for visualization, not for training decisions
    final_history = final_trainer.train(train_final_loader, val_loader=monitor_val_loader)
    
    # Create the final test loader
    test_final_loader = DataLoader(
        test_final_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=MTLBatchCollator(),
        drop_last=False
    )

    # Evaluate on the unseen portion of test set only once
    print("\nEvaluating final model on unseen test data...")
    test_loss, test_metrics = final_trainer.evaluate_with_loss(test_final_loader)
    
    # Plot metrics for final model
    final_trainer.plot_metrics(test_metrics, "Final Model Performance on Test Set", "_final")
    final_trainer.plot_training_curves(final_history, "_final")

    # Print detailed test metrics
    print(f"Test Loss: {test_loss:.4f}")
    valid_test_roc_aucs = []
    print("Test ROC AUC by endpoint:")
    for target in config.TARGET_COLUMNS:
        roc_auc = test_metrics[target]['roc_auc']
        if not math.isnan(roc_auc):
            valid_test_roc_aucs.append(roc_auc)
            print(f"  {target}: ROC AUC={roc_auc:.4f}")
            
    # Calculate and print average
    if valid_test_roc_aucs:
        avg_test_roc_auc = np.mean(valid_test_roc_aucs)
        print(f"Average ROC AUC Across All Tasks: {avg_test_roc_auc:.4f}")

    # Save final model
    save_dir = os.path.dirname('toxicity_prediction_model.pt')
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(final_model.state_dict(), 'toxicity_prediction_model.pt')
    with open('data_processor.pkl', 'wb') as f:
        pickle.dump({'scaler': data_processor.scaler, 'pca': data_processor.pca}, f)
    print("\nThe model has been saved as 'toxicity_prediction_model.pt'")
    print("Data processor state has been saved as 'data_processor.pkl'")


if __name__ == '__main__':
    main()