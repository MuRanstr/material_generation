import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import os
import glob
from project.data_utils import SimpleCrystalData
from project.preprocess import cif_to_training_data

# ----------------------------------------------------------------------
# 1. 晶体数据集类
# ----------------------------------------------------------------------
class CrystalDataset(Dataset):
    """
    用于加载 CIF 文件并将其转换为 SimpleCrystalData 训练样本的数据集。
    """
    def __init__(self, root_dir: str, max_t: float = 1000.0):
        """
        Args:
            root_dir: 包含 CIF 文件的根目录。
            max_t: 最大的噪声时间步。
        """
        self.cif_files = glob.glob(os.path.join(root_dir, "*.cif"))
        if not self.cif_files:
            raise FileNotFoundError(f"在目录 {root_dir} 中未找到任何 .cif 文件。")
        
        self.max_t = max_t
        print(f"✅ 成功找到 {len(self.cif_files)} 个 CIF 文件。")

    def __len__(self):
        return len(self.cif_files)

    def __getitem__(self, idx: int) -> SimpleCrystalData:
        cif_path = self.cif_files[idx]
        # 随机生成一个时间步 t，模拟扩散模型的训练过程
        t = torch.rand(1).item() * self.max_t
        
        # 使用预处理函数将 CIF 转换为训练样本
        return cif_to_training_data(cif_path, t=t)

# ----------------------------------------------------------------------
# 2. 批次数据处理函数 (Collate Function)
# ----------------------------------------------------------------------
def collate_fn(data_list: List[SimpleCrystalData]) -> SimpleCrystalData:
    """
    将 SimpleCrystalData 列表合并为一个批次 (Batch)。
    """
    if not data_list:
        return None

    # 1. 坐标和原子类型：需要拼接
    pos_list = [data.pos for data in data_list]
    atomic_numbers_list = [data.atomic_numbers for data in data_list]
    
    # 2. 晶格和时间步：不需要拼接，但需要堆叠
    cell_list = [data.cell for data in data_list]
    t_list = [data.t for data in data_list]
    
    # 3. 目标更新量：需要拼接或堆叠
    target_coord_update_list = [data.target_coord_update for data in data_list]
    target_cell_update_list = [data.target_cell_update for data in data_list]
    
    # 4. 批次索引 (Batch Index)：记录每个原子属于哪个晶体
    batch_list = []
    num_atoms_list = []
    current_batch_idx = 0
    for i, pos in enumerate(pos_list):
        batch_list.append(torch.full((pos.shape[0],), i, dtype=torch.long))
        num_atoms_list.append(pos.shape[0])
    
    # 5. 合并张量
    return SimpleCrystalData(
        pos=torch.cat(pos_list, dim=0),
        cell=torch.cat(cell_list, dim=0),
        atomic_numbers=torch.cat(atomic_numbers_list, dim=0),
        batch=torch.cat(batch_list, dim=0),
        num_atoms=torch.tensor(num_atoms_list, dtype=torch.long),
        t=torch.cat(t_list, dim=0),
        target_coord_update=torch.cat(target_coord_update_list, dim=0),
        target_cell_update=torch.cat(target_cell_update_list, dim=0),
    )

# ----------------------------------------------------------------------
# 3. 主运行逻辑 (测试)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 假设 CIF 文件在 generated_cifs 目录下
    CIF_DIR = "../data"
    
    try:
        dataset = CrystalDataset(root_dir=CIF_DIR)
        
        # 使用 DataLoader 进行批次加载
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        print(f"✅ DataLoader 创建成功，批次大小: 4")
        
        # 迭代第一个批次
        for i, batch in enumerate(dataloader):
            if i == 0:
                print("\n--- 第一个批次数据信息 ---")
                print(f"批次中晶体数量: {batch.cell.shape[0]}")
                print(f"批次中原子总数: {batch.pos.shape[0]}")
                print(f"时间步 t 形状: {batch.t.shape}")
                print(f"坐标更新目标形状: {batch.target_coord_update.shape}")
                print(f"晶格更新目标形状: {batch.target_cell_update.shape}")
                break
        
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print("请确保 'generated_cifs' 目录下有 CIF 文件，或修改 CIF_DIR 路径。")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
