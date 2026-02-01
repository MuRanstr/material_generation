import torch
from dataclasses import dataclass


@dataclass
class SimpleCrystalData:
    pos: torch.Tensor          # 原子分数坐标 (N_atoms, 3)
    cell: torch.Tensor         # 晶格矩阵 (N_crystals, 3, 3)
    atomic_numbers: torch.Tensor # 原子类型 (N_atoms,)
    batch: torch.Tensor        # 批次索引 (N_atoms,)
    num_atoms: torch.Tensor    # 每个晶体的原子数 (N_crystals,)
    # 训练时使用
    t: torch.Tensor = torch.tensor([0.0]) # 时间步 t
    target_coord_update: torch.Tensor = torch.tensor([0.0]) # 目标坐标更新
    target_cell_update: torch.Tensor = torch.tensor([0.0]) # 目标晶格更新

