import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
import os
import glob
import json
from typing import List, Dict, Tuple
from project.models.diffusion_model import SimpleMatterGen
from project.data_utils import SimpleCrystalData
from project.dataset.material_dataset import CrystalDataset
from project.utils.geo_utils import custom_evaluate_material
from project.generate_stable_materials import Structure


class ValueNetwork(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.t_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 特征提取层
        self.atom_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: SimpleCrystalData):
        t_expanded = x.t.unsqueeze(-1)
        t_embedding = self.t_encoder(t_expanded)

        atom_features = torch.ones(x.pos.shape[0], self.hidden_dim, device=x.pos.device)
        t_per_atom = t_embedding[x.batch]
        h = atom_features * t_per_atom
        h = self.atom_mlp(h)

        h_crystal = torch.zeros(x.cell.shape[0], self.hidden_dim, device=x.pos.device).index_add_(
            0, x.batch, h
        )

        return self.value_head(h_crystal)


class PPOOptimizer:
    def __init__(
            self,
            model: SimpleMatterGen,
            lr_actor: float = 1e-4,
            lr_critic: float = 2e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            eps_clip: float = 0.2,
            k_epochs: int = 10,
            sigma_init: float = 0.1
    ):
        self.actor = model
        self.critic = ValueNetwork(model.hidden_dim).to(next(model.parameters()).device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        # 动作空间的标准差
        self.log_sigma = nn.Parameter(torch.ones(1) * np.log(sigma_init))
        self.optimizer_actor.add_param_group({'params': [self.log_sigma]})

    def select_action(self, data: SimpleCrystalData) -> Dict:
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            predictions = self.actor(data, data.t)

            coord_mean = predictions["coord_update"]
            coord_dist = Normal(coord_mean, torch.exp(self.log_sigma))
            coord_action = coord_dist.sample()
            coord_log_prob = coord_dist.log_prob(coord_action).sum()

            cell_mean = predictions["cell_update"]
            cell_dist = Normal(cell_mean, torch.exp(self.log_sigma))
            cell_action = cell_dist.sample()
            cell_log_prob = cell_dist.log_prob(cell_action).sum()

            atom_logits = predictions["atom_type_logits"]
            atom_dist = Categorical(logits=atom_logits)
            atom_action = atom_dist.sample()
            atom_log_prob = atom_dist.log_prob(atom_action).sum()

            state_value = self.critic(data)

        return {
            "coord": coord_action,
            "cell": cell_action,
            "atom": atom_action,
            "log_prob": coord_log_prob + cell_log_prob + atom_log_prob,
            "state_value": state_value.squeeze()
        }

    def compute_gae(self, rewards: List[float], values: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] - values[i].item()
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.stack(values).detach().cpu()

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        return advantages, returns

    def update(self, states: List[SimpleCrystalData], actions: List[Dict], rewards: List[float],
               old_log_probs: List[torch.Tensor], values: List[torch.Tensor]):
        advantages, returns = self.compute_gae(rewards, values)

        device = next(self.actor.parameters()).device
        advantages = advantages.to(device)
        returns = returns.to(device)
        old_log_probs = torch.stack(old_log_probs).detach()

        actor_losses = []
        critic_losses = []

        for _ in range(self.k_epochs):
            self.actor.train()
            self.critic.train()

            # 1. 计算所有状态的损失
            total_actor_loss = 0
            total_critic_loss = 0

            for i, data in enumerate(states):
                # Actor
                predictions = self.actor(data, data.t)
                coord_dist = Normal(predictions["coord_update"], torch.exp(self.log_sigma))
                cell_dist = Normal(predictions["cell_update"], torch.exp(self.log_sigma))
                atom_dist = Categorical(logits=predictions["atom_type_logits"])

                new_lp = (coord_dist.log_prob(actions[i]["coord"]).sum() +
                          cell_dist.log_prob(actions[i]["cell"]).sum() +
                          atom_dist.log_prob(actions[i]["atom"]).sum())

                ratio = torch.exp(new_lp - old_log_probs[i])
                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[i]
                total_actor_loss += -torch.min(surr1, surr2)

                # Critic
                val = self.critic(data)
                total_critic_loss += nn.MSELoss()(val.squeeze(), returns[i])

            # 2. 平均损失
            avg_actor_loss = total_actor_loss / len(states)
            avg_critic_loss = total_critic_loss / len(states)

            # 3. 统一更新
            self.optimizer_actor.zero_grad()
            avg_actor_loss.backward()
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            avg_critic_loss.backward()
            self.optimizer_critic.step()

            actor_losses.append(avg_actor_loss.item())
            critic_losses.append(avg_critic_loss.item())

        return np.mean(actor_losses), np.mean(critic_losses)


def get_reward_info(cif_path: str):
    """
    评估材料并返回详细的奖励信息
    """
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        try:
            custom_evaluate_material(cif_path)
            output = f.getvalue()
        except Exception as e:
            return -10.0, None, 0.0, False

    delta_g = None
    stability_score = 0.0
    passed = False

    for line in output.split('\n'):
        if "Avg HER ΔG (eV):" in line:
            try:
                val = line.split(':')[1].strip().split(' ')[0]
                if val != 'N/A' and val != 'Error':
                    delta_g = float(val)
            except:
                pass
        if "Stability Score:" in line:
            try:
                stability_score = float(line.split(':')[1].split('(')[0].strip())
            except:
                pass
        if "是否通过筛选: 是" in line:
            passed = True

    if delta_g is None:
        return -5.0, None, stability_score, passed

    reward = np.exp(-(delta_g ** 2) / (2 * 0.1 ** 2)) * 10.0
    if passed:
        reward += 5.0
    else:
        reward -= 2.0

    return reward, delta_g, stability_score, passed


def save_as_cif(action, cif_source_path, output_path):
    try:
        struct = Structure.from_file(cif_source_path)
        coord_updates = action["coord"].cpu().numpy()
        for i, site in enumerate(struct):
            if i < len(coord_updates):
                new_coords = site.frac_coords + coord_updates[i]
                struct.replace(i, site.species, coords=new_coords)
        struct.to(filename=output_path)
        return True
    except Exception as e:
        print(f"Structure Update Error: {e}")
        return False


def finetune_rl():
    model = SimpleMatterGen(hidden_dim=128)
    if os.path.exists("mattergen_model.pth"):
        model.load_state_dict(torch.load("mattergen_model.pth"))
        print("已加载预训练权重。")

    ppo = PPOOptimizer(model)
    dataset_dir = "../data"
    dataset = CrystalDataset(root_dir=dataset_dir)
    cif_files = glob.glob(os.path.join(dataset_dir, "*.cif"))

    num_episodes = 100
    print(f"\n--- 开始标准 PPO 强化学习微调 ({num_episodes} Episodes) ---")

    loss_history = []

    for episode in range(num_episodes):
        states, actions, log_probs, rewards, values = [], [], [], [], []

        for i in range(min(len(dataset), 5)):
            data = dataset[i]
            action_info = ppo.select_action(data)
            source_cif = cif_files[i]
            temp_cif = f"temp_rl_{episode}_{i}.cif"
            if save_as_cif(action_info, source_cif, temp_cif):
                reward, dg, ss, ps = get_reward_info(temp_cif)
                states.append(data)
                actions.append(action_info)
                log_probs.append(action_info["log_prob"])
                rewards.append(reward)
                values.append(action_info["state_value"])
                if os.path.exists(temp_cif):
                    os.remove(temp_cif)

        if states:
            actor_loss, critic_loss = ppo.update(states, actions, rewards, log_probs, values)
            avg_reward = sum(rewards) / len(rewards)
            print(
                f"Episode {episode + 1}/{num_episodes} | Avg Reward: {avg_reward:.4f} | Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f}")

            loss_history.append({
                "episode": episode + 1,
                "actor_loss": float(actor_loss),
                "critic_loss": float(critic_loss),
                "avg_reward": float(avg_reward)
            })

    # 保存训练历史
    with open("rl_training_history.json", "w") as f:
        json.dump(loss_history, f, indent=4)
    print("训练历史已保存至 rl_training_history.json")

    torch.save(model.state_dict(), "mattergen_model_rl_tuned.pth")
    print("\n标准 RL 微调完成，模型已保存至 mattergen_model_rl_tuned.pth")


if __name__ == "__main__":
    finetune_rl()