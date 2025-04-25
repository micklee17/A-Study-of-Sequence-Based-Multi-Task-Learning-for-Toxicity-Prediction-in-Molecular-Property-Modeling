import torch
import torch.nn as nn
from config import Config

class TaskSpecificAttention(nn.Module):
    # 新增：任务特定的注意力模块（MTAN核心）
    def __init__(self, embed_dim, attention_dim, dropout):
        super(TaskSpecificAttention, self).__init__()
        self.query = nn.Linear(embed_dim, attention_dim)
        self.key = nn.Linear(embed_dim, attention_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(attention_dim, dtype=torch.float32))

    def forward(self, x):
        # x: (batch_size, embed_dim)
        batch_size = x.size(0)
        # 扩展维度以模拟序列长度为1
        x = x.unsqueeze(1)  # (batch_size, 1, embed_dim)
        Q = self.query(x)   # (batch_size, 1, attention_dim)
        K = self.key(x)     # (batch_size, 1, attention_dim)
        V = self.value(x)   # (batch_size, 1, embed_dim)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attended = torch.matmul(attention_weights, V)  # (batch_size, 1, embed_dim)
        return attended.squeeze(1)  # (batch_size, embed_dim)

class MultiTaskTransformer(nn.Module):
    def __init__(self, config: Config, data_processor):
        super(MultiTaskTransformer, self).__init__()
        self.config = config
        self.data_processor = data_processor
        self.attention = nn.MultiheadAttention(
            embed_dim=config.EMBED_SIZE,
            num_heads=config.NUM_HEADS,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(config.EMBED_SIZE)
        self.norm2 = nn.LayerNorm(config.EMBED_SIZE)
        self.ff = nn.Sequential(
            nn.Linear(config.EMBED_SIZE, config.FF_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FF_DIM, config.EMBED_SIZE)
        )
        self.dropout = nn.Dropout(config.DROPOUT)

        self.shared_layer = nn.Sequential(
            nn.Linear(config.EMBED_SIZE , config.HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(config.DROPOUT)
        )
        # 新增：任务特定的注意力模块（MTAN）
        self.task_attention = nn.ModuleDict({
            task: TaskSpecificAttention(
                embed_dim=config.HIDDEN_SIZE,
                attention_dim=config.ATTENTION_DIM,
                dropout=config.DROPOUT
            ) for task in config.TARGET_COLUMNS
        })
        self.task_layers = nn.ModuleDict({
            task: nn.Linear(config.HIDDEN_SIZE, 1) for task in config.TARGET_COLUMNS
        })

    def forward(self, smiles_list):
        # Get token-level embeddings from ChemBERTa 
        token_embeddings = self.data_processor.smiles_to_embedding(smiles_list)
        
        # Apply mean pooling to get molecule-level embeddings
        # (batch_size, seq_len, embed_size) -> (batch_size, embed_size)
        x_pooled = torch.mean(token_embeddings, dim=1)
        
        # Apply shared layer to transform dimensions
        # (batch_size, embed_size) -> (batch_size, hidden_size)
        shared_features = self.shared_layer(x_pooled)
        
        # Now apply task-specific attention modules
        outputs = []
        for task in self.task_layers.keys():
            task_features = self.task_attention[task](shared_features)
            task_output = self.task_layers[task](task_features)
            outputs.append(task_output)
        return torch.cat(outputs, dim=1)

    def get_l2_regularization(self):
        l2_reg = torch.tensor(0.0, device=next(self.parameters()).device)
        # 共享层的正则化
        for name, param in self.shared_layer.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param, p=2) ** 2
        # 新增：任务注意力模块的正则化
        for task in self.task_attention:
            for name, param in self.task_attention[task].named_parameters():
                if 'weight' in name:
                    l2_reg += torch.norm(param, p=2) ** 2
        return l2_reg