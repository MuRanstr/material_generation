import torch
import torch.nn as nn
import torch.optim as optim
from project.models.diffusion_model import SimpleMatterGen
from project.dataset.material_dataset import CrystalDataset, collate_fn
from torch.utils.data import DataLoader
import os
import glob
import warnings
import json

warnings.filterwarnings("ignore")


def train_mattergen(model: SimpleMatterGen, dataloader: DataLoader, num_epochs: int = 100, lr: float = 1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 损失函数
    coord_loss_fn = nn.MSELoss()
    cell_loss_fn = nn.MSELoss()
    atom_type_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    print(f"\n--- 开始扩散模型训练 ({num_epochs} Epochs) ---")

    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_coord_loss = 0.0
        total_cell_loss = 0.0
        total_atom_loss = 0.0
        num_batches = 0

        for data in dataloader:
            optimizer.zero_grad()

            # 1. 前向传播
            predictions = model(data, data.t)

            # 2. 计算损失
            coord_loss = coord_loss_fn(predictions["coord_update"], data.target_coord_update)

            target_cell_update_batch = data.target_cell_update.view(-1, 3, 3)
            target_cell_params = torch.cat([
                target_cell_update_batch[:, 0, 0].unsqueeze(1),
                target_cell_update_batch[:, 1, 1].unsqueeze(1),
                target_cell_update_batch[:, 2, 2].unsqueeze(1),
                torch.zeros(data.cell.shape[0], 3, device=data.cell.device)
            ], dim=1)

            cell_loss = cell_loss_fn(predictions["cell_update"], target_cell_params)
            atom_type_loss = atom_type_loss_fn(predictions["atom_type_logits"], data.atomic_numbers)

            # 组合损失 (带有权重平衡)
            loss = coord_loss * 10.0 + cell_loss * 1.0 + atom_type_loss * 0.1

            # 3. 反向传播与优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_coord_loss += coord_loss.item()
            total_cell_loss += cell_loss.item()
            total_atom_loss += atom_type_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_coord = total_coord_loss / num_batches
        avg_cell = total_cell_loss / num_batches
        avg_atom = total_atom_loss / num_batches

        loss_history.append({
            "epoch": epoch + 1,
            "loss": float(avg_loss),
            "coord_loss": float(avg_coord),
            "cell_loss": float(avg_cell),
            "atom_loss": float(avg_atom)
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.6f}, Coord: {avg_coord:.6f}, Cell: {avg_cell:.6f}, Atom: {avg_atom:.6f}")

    # 保存损失历史
    with open("models/loss_history.json", "w") as f:
        json.dump(loss_history, f, indent=4)
    print("扩散模型训练损失历史已保存至 loss_history.json")

    torch.save(model.state_dict(), "models/mattergen_model.pth")
    print("模型权重已保存至 mattergen_model.pth")



if __name__ == "__main__":
    model = SimpleMatterGen(hidden_dim=128)
    CIF_DATA_DIR = "data"

    if not os.path.exists(CIF_DATA_DIR) or not glob.glob(os.path.join(CIF_DATA_DIR, "*.cif")):
        print(f"错误：未找到 CIF 文件。请将您的 CIF 文件放入 {CIF_DATA_DIR} 目录。")
        import sys

        sys.exit(1)

    dataset = CrystalDataset(root_dir=CIF_DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    print(f"成功创建 DataLoader，包含 {len(dataset)} 个样本。")

    # 演示目的，运行较少 epoch
    train_mattergen(model, dataloader, num_epochs=20, lr=1e-3)