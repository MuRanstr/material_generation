import torch
import torch.nn as nn
from typing import Dict
from project.data_utils import SimpleCrystalData # 导入数据结构

# GNN
class SimpleEquivariantGNN(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 模拟原子特征处理 (MLP 替代复杂的 GNN)
        self.atom_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 预测坐标的去噪量
        self.coord_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3) # 输出 3D 坐标的更新量
        )
        
        # 预测晶格的去噪量
        self.cell_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 6) # 6个晶格参数的变化量
        )
        
        # 预测原子类型的 Logits
        self.MAX_ATOMIC_NUM = 100
        self.atom_type_predictor = nn.Linear(hidden_dim, self.MAX_ATOMIC_NUM)

    def forward(self, x: SimpleCrystalData, t_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x.pos: (N_atoms, 3)
        # x.cell: (N_crystals, 3, 3)
        # x.batch: (N_atoms,)
        # t_embedding: (N_crystals, hidden_dim)
        
        # 1. 构造原子特征
        atom_features = torch.ones(x.pos.shape[0], self.hidden_dim, device=x.pos.device)
        t_per_atom = t_embedding[x.batch]
        h = atom_features * t_per_atom
        
        # 2. 模拟 GNN 聚合和特征更新
        h = self.atom_mlp(h) # (N_atoms, hidden_dim)
        
        # 3. 预测去噪结果
        pred_coord_update = self.coord_predictor(h)
        
        # 聚合原子特征到晶体级别
        h_crystal = torch.zeros(x.cell.shape[0], self.hidden_dim, device=x.pos.device).index_add_(
            0, x.batch, h
        )
        pred_cell_update = self.cell_predictor(h_crystal)
        pred_atom_type_logits = self.atom_type_predictor(h)
        
        return {
            "coord_update": pred_coord_update,
            "cell_update": pred_cell_update,
            "atom_type_logits": pred_atom_type_logits,
        }

# 扩散模型
class SimpleMatterGen(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.T = 1000 # 假设总时间步
        
        # 噪声水平编码
        self.t_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 核心去噪网络
        self.denoiser = SimpleEquivariantGNN(hidden_dim)
        
    def forward(self, x: SimpleCrystalData, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        t_expanded = t.unsqueeze(-1)
        t_embedding = self.t_encoder(t_expanded)
        predictions = self.denoiser(x, t_embedding)
        return predictions
