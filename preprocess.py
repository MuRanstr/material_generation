import torch
from pymatgen.core import Structure
from typing import Optional
import random
import os
from project.data_utils import SimpleCrystalData

# 1. CIF 文件预处理函数

def cif_to_training_data(cif_path: str, t: Optional[float] = None) -> SimpleCrystalData:
    """
    将单个 CIF 文件转换为 SimpleCrystalData 格式的训练样本。
    
    Args:
        cif_path: CIF 文件路径。
        t: 噪声时间步 (0-1000)。如果为 None，则随机生成。
        
    Returns:
        SimpleCrystalData 训练样本。
    """
    # 1. 读取 CIF 文件并创建 pymatgen Structure 对象
    try:
        structure = Structure.from_file(cif_path)
    except Exception as e:
        raise ValueError(f"无法读取或解析 CIF 文件 {cif_path}: {e}")

    # 2. 提取目标结构 (x_0) 的特征
    num_atoms = len(structure)
    
    # 坐标 (分数坐标)
    pos_0 = torch.tensor(structure.frac_coords, dtype=torch.float32)
    
    # 晶格 (3x3 矩阵)
    cell_0 = torch.tensor(structure.lattice.matrix, dtype=torch.float32).unsqueeze(0)
    
    # 原子类型 (原子序数)
    atomic_numbers_0 = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
    
    # 批次信息
    batch = torch.zeros(num_atoms, dtype=torch.long)
    num_atoms_tensor = torch.tensor([num_atoms], dtype=torch.long)
    
    # 3. 随机时间步 t
    t_float = t if t is not None else random.random() * 1000
    t_tensor = torch.tensor([t_float])
    
    # 4. 模拟加噪过程 (x_t)
    # 噪声强度与 t 成正比
    noise_scale = t_float / 1000.0
    
    # 坐标加噪
    noise_pos = torch.randn_like(pos_0) * noise_scale * 0.1
    pos_t = pos_0 + noise_pos
    
    # 晶格加噪
    noise_cell = torch.randn_like(cell_0) * noise_scale * 0.05
    cell_t = cell_0 + noise_cell
    
    # 5. 计算目标去噪量 (Target Update)
    # MatterGen 预测的是 score/epsilon，这里简化为预测 x_0 - x_t
    target_coord_update = pos_0 - pos_t
    target_cell_update = cell_0 - cell_t
    
    # 6. 封装为 SimpleCrystalData
    return SimpleCrystalData(
        pos=pos_t, 
        cell=cell_t, 
        atomic_numbers=atomic_numbers_0, # 目标是预测正确的原子类型
        batch=batch, 
        num_atoms=num_atoms_tensor,
        t=t_tensor,
        target_coord_update=target_coord_update.squeeze(0),
        target_cell_update=target_cell_update.squeeze(0),
    )

# ----------------------------------------------------------------------
# 2. 主运行逻辑 (测试)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 假设我们有一个 CIF 文件，使用之前生成的示例文件进行测试
    TEST_CIF_PATH = "./generated_cifs/generated_structure_15atoms.cif"
    
    if not os.path.exists(TEST_CIF_PATH):
        print(f"错误：未找到测试文件 {TEST_CIF_PATH}。请先运行 generate.py 生成该文件。")
    else:
        print(f"找到测试文件 {TEST_CIF_PATH}，开始转换...")
        
        # 转换
        training_sample = cif_to_training_data(TEST_CIF_PATH, t=500.0)
        
        # 打印转换结果
        print("\n--- 转换结果 (SimpleCrystalData) ---")
        print(f"原子总数: {training_sample.pos.shape[0]}")
        print(f"时间步 t: {training_sample.t.item():.2f}")
        print(f"带噪坐标 (pos_t) 形状: {training_sample.pos.shape}")
        print(f"目标坐标更新 (target_coord_update) 形状: {training_sample.target_coord_update.shape}")
        print(f"目标原子类型 (atomic_numbers) 示例: {training_sample.atomic_numbers[:5]}")
        print(f"带噪晶格 (cell_t) 形状: {training_sample.cell.shape}")
        print(f"目标晶格更新 (target_cell_update) 形状: {training_sample.target_cell_update.shape}")
        
        # 验证目标去噪量是否接近 t=500 时的预期噪声水平
        coord_update_norm = training_sample.target_coord_update.norm().item()
        print(f"目标坐标更新的 L2 范数: {coord_update_norm:.4f}")
        
        # 验证晶格更新量是否合理
        cell_update_norm = training_sample.target_cell_update.norm().item()
        print(f"目标晶格更新的 L2 范数: {cell_update_norm:.4f}")
        
        print("\n转换成功，该对象可直接用于 train.py 中的训练循环。")
