import random
from typing import List, Optional
import pandas as pd
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer, AutoModel
from config import Config
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from pathlib import Path

def enumerate_smiles(smiles, num_variants=10, max_attempts=30):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles] * num_variants

    variants = set()
    attempts = 0

    while len(variants) < num_variants and attempts < max_attempts:
        new_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True, kekuleSmiles=True)
        new_mol = Chem.MolFromSmiles(new_smiles)
        if new_mol is not None and new_smiles not in variants:
            variants.add(new_smiles)
        attempts += 1

    # 如果未能生成足够的变体，用原始SMILES填充列表
    variants = list(variants)
    while len(variants) < num_variants:
        variants.append(smiles)

    return variants[:num_variants]

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler() if config.MIN_MAX_SCALE else None
        self.pca = PCA(n_components=config.EMBED_SIZE) if config.FEATURE_COMPRESS else None
        self.model = None
        self.tokenizer = None

    def load_pretrained_model(self):
        """Load pretrained model based on configuration choice"""
        local_model_path = "./chemberta"  # Use relative path from the script location
        
        model_names = {
            1: "seyonec/ChemBERTa-zinc-base-v1",
            2: "pchanda/pretrained-smiles-pubchem10m",
            3: "HUBioDataLab/SELFormer"
        }
        
        try:
            # Ensure model files exist locally
            model_name = model_names.get(self.config.PRETRAIN_CHOICE)
            if not model_name:
                raise ValueError("Invalid pre-trained model selection")
                
            local_path = Path(local_model_path)
            required_files = ["config.json", "pytorch_model.bin"]
            
            if local_path.exists() and all((local_path / file).exists() for file in required_files):
                print(f"Loading model from local path: {local_path}")
                self.model = AutoModel.from_pretrained(str(local_path))
                self.tokenizer = self.config.TOKENIZER or AutoTokenizer.from_pretrained(str(local_path))
            else:
                print(f"Model files missing. Loading model from {model_name}")
                self.model = AutoModel.from_pretrained(model_name) 
                self.tokenizer = self.config.TOKENIZER or AutoTokenizer.from_pretrained(model_name)
                
                # Save model for future use
                if not local_path.exists():
                    local_path.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(local_path)
                self.tokenizer.save_pretrained(local_path)
                print(f"Model and tokenizer saved to {local_path}")
                
        except Exception as e:
            print(f"Error loading from local path: {e}")
            print(f"Falling back to loading from {model_name}")
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer = self.config.TOKENIZER or AutoTokenizer.from_pretrained(model_name)
            
        # Print parameter information
        self.check_model_params()
        
        # Set model to training mode to enable gradient flow
        self.model.train()

    def check_model_params(self):
        """Check and print model parameter information"""
        trainable_params = 0
        total_params = 0
        
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
        
        print("\n===== MODEL PARAMETERS =====")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100.0 * trainable_params / total_params:.2f}%)")
        print("===========================")

    def smiles_to_embedding(self, smiles_list: List[str]):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call load_pretrained_model first.")
        try:
            inputs = self.tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True,
                                    max_length=self.config.MAX_POSITION_EMBEDDINGS)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.to(dtype=torch.float32)
        except Exception as e:
            print(f"Error generating embedding for SMILES: {e}")
            return torch.zeros((len(smiles_list), self.config.MAX_POSITION_EMBEDDINGS, self.config.EMBED_SIZE), dtype=torch.float32)

    def smiles_to_fingerprint(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(167, dtype=np.float32)
        fingerprint = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((167,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, arr)
        return arr.astype(np.float32)

    def preprocess_data(self, df, is_train=True):

        na_count = df.iloc[:, 0:13].isna().sum(axis=1)
        df = df[na_count < 7]

        # Filter valid SMILES
        df['valid_smiles'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)
        df = df[df['valid_smiles']].copy().reset_index(drop=True)
        df = df.drop(columns=['valid_smiles'], errors='ignore')

        # Standardize SMILES
        def standardize_smiles(smiles):
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol) if mol else None

        df['smiles'] = df['smiles'].apply(standardize_smiles)
        # Remove duplicates based on 'smiles' column
        df = df.drop_duplicates(subset='smiles').reset_index(drop=True)

        # Drop any rows with invalid SMILES after standardization
        df = df.dropna(subset=['smiles']).reset_index(drop=True)

        y = df.fillna(-1)[self.config.TARGET_COLUMNS].values.astype(np.float32)

        return y

class SmilesFinetuneDataset(Dataset):
    def __init__(self, data: pd.DataFrame, task_names: List, tokenizer, data_processor,
                 max_length: int = Config.MAX_POSITION_EMBEDDINGS,
                 num_variants: int = Config.NUM_SMILES_VARIANTS, enumerate_boo: bool = False):
        self.data = data
        self.task_names = task_names
        self.tokenizer = tokenizer
        self.data_processor = data_processor
        self.max_length = max_length
        self.num_variants = num_variants
        self.enumerate_boo = enumerate_boo
        if set(self.task_names) != set(Config.TARGET_COLUMNS):
            raise ValueError("task_names must match Config.TARGET_COLUMNS")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        smiles = row["smiles"]
        labels = [float(row[task]) if not pd.isna(row[task]) else -1 for task in self.task_names]
        smiles_variants = [smiles]

        if self.enumerate_boo:
            try:
                # FIX: Swap parameters to correct order (num_variants first, max_attempts second)
                smiles_variants = enumerate_smiles(
                    smiles, 
                    self.num_variants,  # Correctly pass as num_variants
                    Config.MAX_SMILES_VARIANTS_ATTEMPTS  # Correctly pass as max_attempts
                )
                smiles = random.choice(smiles_variants)
            except Exception as e:
                print(f"Error enumerating SMILES: {e}")

        # Generate fingerprint if using feature fusion
        # fingerprint = self.data_processor.smiles_to_fingerprint(smiles)
        # fingerprint_tensor = torch.tensor(fingerprint, dtype=torch.float32)

        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return {
            "smiles": smiles,
            "labels": labels_tensor,
            # "fingerprint": fingerprint_tensor,  # Uncomment if using fingerprints
            "smiles_variants": smiles_variants if self.enumerate_boo else None,
        }
