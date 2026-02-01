import json
import os
import glob
import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout

# 导入评估函数
from geo_utils import custom_evaluate_material


def plot_diffusion_training_loss(history_path="../models/loss_history.json"):
    """
    1. 绘制扩散模型的训练损失图
    """
    if not os.path.exists(history_path):
        print(f"⚠️ 未找到 {history_path}，跳过扩散模型损失绘图。")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    total_loss = [h["loss"] for h in history]
    coord_loss = [h["coord_loss"] for h in history]
    cell_loss = [h["cell_loss"] for h in history]
    atom_loss = [h["atom_loss"] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_loss, label="Total Loss", linewidth=2, color='black')
    plt.plot(epochs, coord_loss, label="Coord Loss (weighted)", linestyle="--", alpha=0.7)
    plt.plot(epochs, cell_loss, label="Cell Loss", linestyle="--", alpha=0.7)
    plt.plot(epochs, atom_loss, label="Atom Loss", linestyle="--", alpha=0.7)

    plt.title("Diffusion Model Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")  # 使用对数刻度以便观察细节
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)

    plt.savefig("../results/loss_curve.png")
    print("✅ 扩散模型训练损失图已保存至 loss_curve.png")


def plot_rl_training_results(history_path="../models/rl_training_history.json"):
    """
    2. 绘制强化学习的训练损失和奖励趋势
    """
    if not os.path.exists(history_path):
        print(f"⚠️ 未找到 {history_path}，跳过 RL 训练图。")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    episodes = [h["episode"] for h in history]
    actor_loss = [h["actor_loss"] for h in history]
    critic_loss = [h["critic_loss"] for h in history]
    avg_reward = [h["avg_reward"] for h in history]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(episodes, actor_loss, label='Actor Loss', color='tab:red', marker='o')
    ax1.plot(episodes, critic_loss, label='Critic Loss', color='tab:orange', marker='s')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Avg Reward', color=color)
    ax2.plot(episodes, avg_reward, label='Avg Reward', color='tab:blue', marker='^', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title("RL Training Progress: Loss and Reward")
    fig.tight_layout()
    plt.grid(True, alpha=0.3)

    plt.savefig("../results/rl_loss_curve.png")
    print("✅ 强化学习训练进度图已保存至 rl_loss_curve.png")


def collect_and_plot_distributions(cif_dir="generated_cifs"):
    """
    3. 绘制 deltaG 和 Stability 分布图
    """
    cif_files = glob.glob(os.path.join(cif_dir, "*.cif"))
    if not cif_files:
        print(f"⚠️ 在 {cif_dir} 中未找到 CIF 文件，跳过分布图。")
        return

    print(f"开始评估 {len(cif_files)} 个晶体结构以获取性能分布...")
    deltags = []
    stabilities = []

    for cif in cif_files:
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                custom_evaluate_material(cif)
                output = f.getvalue()

                for line in output.split('\n'):
                    if "Avg HER ΔG (eV):" in line:
                        val_str = line.split(':')[1].strip().split(' ')[0]
                        if val_str not in ['N/A', 'Error']:
                            deltags.append(float(val_str))
                    if "Stability Score:" in line:
                        try:
                            ss_val = float(line.split(':')[1].split('(')[0].strip())
                            stabilities.append(ss_val)
                        except:
                            pass
            except Exception as e:
                print(f"评估 {cif} 时出错: {e}")

    if deltags:
        plt.figure(figsize=(10, 6))
        plt.hist(deltags, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--', label='Ideal ΔG = 0')
        plt.title("Distribution of ΔG for Generated Crystals")
        plt.xlabel("ΔG (eV)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig("../results/her_performance.png")
        print(f"✅ deltaG 分布图已保存至 her_performance.png")

    if stabilities:
        plt.figure(figsize=(10, 6))
        plt.hist(stabilities, bins=15, color='lightgreen', edgecolor='black', alpha=0.7)
        plt.axvline(x=1.0, color='darkgreen', linestyle='--', label='Perfect Stability = 1.0')
        plt.title("Distribution of Stability Scores for Generated Crystals")
        plt.xlabel("Stability Score (Normalized)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig("../results/stability_curve.png")
        print(f"✅ stability 分布图已保存至 stability_curve.png")


if __name__ == "__main__":
    # 1. 绘制扩散模型训练损失
    plot_diffusion_training_loss()

    # 2. 绘制强化学习训练进度
    plot_rl_training_results()

    # 3. 绘制性能和稳定性分布图
    collect_and_plot_distributions("../last_ten_structure")