import torch
from typing import Tuple
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter
from pymatgen.core.periodic_table import Element
import os
from project.models.diffusion_model import SimpleMatterGen
from data_utils import SimpleCrystalData


def generate_structure(model: SimpleMatterGen, num_atoms: int, num_steps: int = 100, output_dir: str = "./generated_cifs", index: int = 0) -> Tuple[Structure, str]:
    """
    从随机噪声开始，迭代生成一个晶体结构。
    """
    print(f"开始生成结构 #{index} (原子数: {num_atoms}, 步数: {num_steps})")
    
    # 1. 初始化：从随机噪声开始 (t=T)
    T = model.T
    pos = torch.rand(num_atoms, 3)
    cell = torch.eye(3) * 10.0 + torch.rand(3, 3) * 2.0
    cell = cell.unsqueeze(0) # (1, 3, 3)
    atomic_numbers = torch.ones(num_atoms, dtype=torch.long)
    batch = torch.zeros(num_atoms, dtype=torch.long)
    num_atoms_tensor = torch.tensor([num_atoms], dtype=torch.long)
    
    current_data = SimpleCrystalData(pos=pos, cell=cell, atomic_numbers=atomic_numbers, batch=batch, num_atoms=num_atoms_tensor)
    
    # 迭代去噪
    for i in range(num_steps):
        t_float = T * (1 - i / num_steps)
        t = torch.tensor([t_float])
        
        with torch.no_grad():
            predictions = model(current_data, t)
        
        # 更新结构
        step_size = 0.01
        current_data.pos = (current_data.pos + predictions["coord_update"] * step_size) % 1.0
        
        pred_cell_params_update = predictions["cell_update"][0]
        a, b, c, alpha, beta, gamma = Lattice(current_data.cell[0].numpy()).parameters
        
        a = max(1.0, a + pred_cell_params_update[0].item() * step_size)
        b = max(1.0, b + pred_cell_params_update[1].item() * step_size)
        c = max(1.0, c + pred_cell_params_update[2].item() * step_size)
        
        new_lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        current_data.cell[0] = torch.tensor(new_lattice.matrix, dtype=torch.float32)
        
        if i == num_steps - 1:
            probs = torch.softmax(predictions["atom_type_logits"], dim=1)
            atomic_numbers_pred = torch.multinomial(probs, num_samples=1).squeeze(1)
            atomic_numbers_pred = torch.clamp(atomic_numbers_pred, min=1, max=80)
            current_data.atomic_numbers = atomic_numbers_pred


    # 4. 转换为 pymatgen 结构并保存
    final_pos = current_data.pos.cpu().numpy()
    final_cell = current_data.cell[0].cpu().numpy()
    final_atomic_numbers = current_data.atomic_numbers.cpu().numpy()
    
    elements = [Element.from_Z(int(z)).symbol for z in final_atomic_numbers]
    
    lattice = Lattice(final_cell)
    structure = Structure(lattice, elements, final_pos, coords_are_cartesian=False)
    
    os.makedirs(output_dir, exist_ok=True)
    cif_path = os.path.join(output_dir, f"generated_structure_{index}_{num_atoms}atoms.cif")
    CifWriter(structure).write_file(cif_path)
    
    return structure, cif_path


if __name__ == "__main__":
    # 1. 创建模型并加载训练好的权重
    model = SimpleMatterGen(hidden_dim=128)
    
    # 尝试加载权重，如果不存在则使用随机权重
    try:
        model.load_state_dict(torch.load("models/mattergen_model_rl_tuned.pth"))
        print("成功加载训练好的模型权重。")
    except FileNotFoundError:
        print("未找到 mattergen_model.pth，将使用随机初始化的模型进行生成。请先运行 train.py 进行训练。")
    
    model.eval()
    
    # 2. 运行批量生成过程
    OUTPUT_DIR = "generated_cifs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    NUM_SAMPLES = 30 # 批量生成 10 个样本
    print(f"\n--- 开始使用模型进行批量生成测试 (共 {NUM_SAMPLES} 个) ---")
    
    generated_files = []
    for i in range(NUM_SAMPLES):
        try:
            # 随机原子数在 4 到 12 之间
            num_atoms = torch.randint(4, 13, (1,)).item()
            structure, cif_path = generate_structure(model, num_atoms=num_atoms, num_steps=100, output_dir=OUTPUT_DIR, index=i)
            generated_files.append(cif_path)
            print(f"结构 #{i} 已保存: {cif_path} (化学式: {structure.formula})")
        except Exception as e:
            print(f"生成结构 #{i} 时发生错误: {e}")
            
    print(f"\n批量生成完成，共生成 {len(generated_files)} 个 CIF 文件。")
