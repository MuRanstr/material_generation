import os
import glob
import math
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
import warnings


warnings.filterwarnings("ignore")


def plot_structure_on_ax(cif_path, ax):
    """
    在指定的 Matplotlib Axes 对象上绘制晶体结构
    """
    try:
        structure = Structure.from_file(cif_path)

        # 获取原子坐标和元素
        coords = structure.cart_coords
        species = [site.species_string for site in structure]

        # 为不同元素分配颜色
        unique_species = list(set(species))
        colors = plt.cm.get_cmap('tab10', len(unique_species))
        color_map = {s: colors(i) for i, s in enumerate(unique_species)}

        # 绘制原子
        for i, (coord, spec) in enumerate(zip(coords, species)):
            ax.scatter(coord[0], coord[1], coord[2],
                       s=100, color=color_map[spec], edgecolors='black',
                       label=spec if i == species.index(spec) else "")

        # 绘制晶胞边界
        lattice = structure.lattice.matrix
        vertices = []
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    vertices.append(i * lattice[0] + j * lattice[1] + k * lattice[2])

        edges = [
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)
        ]

        for edge in edges:
            v1, v2 = vertices[edge[0]], vertices[edge[1]]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], color='gray', linestyle='--', alpha=0.5)

        # 尝试绘制化学键
        try:
            cnn = CrystalNN()
            for i in range(len(structure)):
                neighbors = cnn.get_nn_info(structure, i)
                for nb in neighbors:
                    j = nb['site_index']
                    if i < j:  # 避免重复绘制
                        c1 = structure[i].coords
                        c2 = nb['site'].coords
                        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], color='black', linewidth=1, alpha=0.6)
        except:
            pass

        ax.set_title(f"{structure.formula}\n({os.path.basename(cif_path)})", fontsize=10)
        ax.set_xlabel("X (Å)", fontsize=8)
        ax.set_ylabel("Y (Å)", fontsize=8)
        ax.set_zlabel("Z (Å)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc='upper right', fontsize=7)

    except Exception as e:
        ax.text(0.5, 0.5, 0.5, f"Error: {e}", transform=ax.transAxes)
        print(f"可视化 {cif_path} 时出错: {e}")


def batch_visualize_to_single_image(cif_dir="../last_ten_structure", output_file="../results/generated_structure.png", max_images=10):
    cif_files = sorted(glob.glob(os.path.join(cif_dir, "*.cif")))
    if not cif_files:
        print(f"⚠在 {cif_dir} 中未找到 CIF 文件。")
        return

    # 限制处理的图片数量
    cif_files = cif_files[:max_images]
    num_files = len(cif_files)

    # 计算行列数 (例如 10 张图可以用 2行5列)
    cols = 5
    rows = math.ceil(num_files / cols)

    # 创建大图
    fig = plt.figure(figsize=(cols * 4, rows * 4))

    for idx, cif in enumerate(cif_files):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        plot_structure_on_ax(cif, ax)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✅ 合并图像已保存至: {output_file}")


if __name__ == "__main__":
    # 执行合并可视化
    batch_visualize_to_single_image()
