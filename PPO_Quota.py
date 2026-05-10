import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
import asyncio
from torch.optim.lr_scheduler import StepLR


class ActorCritic(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.actor = nn.Linear(hidden_dim, 2)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared = self.shared(x)
        action_logits = self.actor(shared)
        action_probs = F.softmax(action_logits, dim=-1)
        state_value = self.critic(shared)
        state_value = state_value.squeeze()
        return action_probs, state_value


class SPGG(nn.Module):
    def __init__(self, L_num, device, alpha, gamma, clip_epsilon, r, epochs, now_time,
                 question, ppo_epochs, batch_size, gae_lambda, output_path, delta, rho,
                 zeta=0.0, beta=2.0):
        super().__init__()
        self.L_num = L_num
        self.device = device
        self.r = r
        self.epochs = epochs
        self.question = question
        self.now_time = now_time

        self.alpha = alpha
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.delta = delta
        self.rho = rho
        self.output_path = output_path

        self.zeta = zeta
        self.beta = beta
        self.quota_limit = int(self.L_num * self.L_num * self.zeta)

        self.policy = ActorCritic().to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=alpha)
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.5)

        self.neibor_kernel = torch.tensor(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]], dtype=torch.float32, device=device
        ).unsqueeze(0).unsqueeze(0)

        self.initial_state = self.init_state(question)
        self.current_state = self.initial_state.clone()

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.shot_tasks = []

    def init_state(self, question):
        if question == 1:  # half-half
            state = torch.zeros(self.L_num, self.L_num)
            state[self.L_num // 2:, :] = 1
        elif question == 2:  # bernoulli
            state = torch.bernoulli(torch.full((self.L_num, self.L_num), 0.5))
        elif question == 3:  # all-defector
            state = torch.zeros(self.L_num, self.L_num)
        elif question == 4:  # all-cooperator
            state = torch.ones(self.L_num, self.L_num)
        return state.to(self.device)

    def encode_state(self, state_matrix):
        # 0:自身策略, 1:邻域合作数, 2:全局合作率
        state_4d = state_matrix.float().unsqueeze(0).unsqueeze(0)  # [1,1,L,L]
        padded = F.pad(state_4d, (1, 1, 1, 1), mode='circular')
        neighbor_coop = F.conv2d(padded, self.neibor_kernel).squeeze()  # [L,L]
        global_coop = torch.mean(state_matrix.float())
        return torch.stack([
            state_matrix.float().squeeze(),
            neighbor_coop,
            global_coop.expand_as(state_matrix)
        ], dim=-1).view(-1, 3)

    def _apply_quota_mechanism(self, state_matrix, reward_matrix):
        """
        完整逻辑：
        1. NC ≤ quota：全奖 C，剩余罚 D
        2. ND ≤ quota：全罚 D，剩余奖 C
        3. 否则：coop_rate > 50% 全罚 D，否则全奖 C
        """
        flat_state = state_matrix.view(-1)
        flat_reward = reward_matrix.view(-1)

        c_indices = (flat_state == 1).nonzero(as_tuple=True)[0]
        d_indices = (flat_state == 0).nonzero(as_tuple=True)[0]

        num_c = int(c_indices.numel())
        num_d = int(d_indices.numel())

        # ========== 论文 2.3 的两个极端情况（优先级最高） ==========
        if num_c <= self.quota_limit:
            # 情况 1：全奖 C，剩余罚 D
            c_quota_target = num_c
            d_quota_target = min(num_d, self.quota_limit - num_c)

        elif num_d <= self.quota_limit:
            # 情况 2：全罚 D，剩余奖 C
            d_quota_target = num_d
            c_quota_target = min(num_c, self.quota_limit - num_d)

        else:
            # 情况 3：你的新需求
            coop_rate = num_c / (num_c + num_d)
            if coop_rate > 0.5:
                # 全罚 D
                d_quota_target = min(num_d, self.quota_limit)
                c_quota_target = 0
            else:
                # 全奖 C
                c_quota_target = min(num_c, self.quota_limit)
                d_quota_target = 0

        # ========== 论文排序逻辑：扶弱抑强 ==========
        def select_target(indices, quota, is_cooperator):
            if quota == 0 or indices.numel() == 0:
                return torch.tensor([], dtype=torch.long, device=self.device)
            if indices.numel() <= quota:
                return indices

            vals = flat_reward[indices]
            if is_cooperator:  # C：收益最低
                _, rel = torch.topk(vals, quota, largest=False)
            else:  # D：收益最高
                _, rel = torch.topk(vals, quota, largest=True)
            return indices[rel]

        target_c = select_target(c_indices, c_quota_target, True)
        target_d = select_target(d_indices, d_quota_target, False)

        reward_adjusted = reward_matrix.view(-1).clone()
        if target_c.numel() > 0:
            reward_adjusted[target_c] += self.beta
        if target_d.numel() > 0:
            reward_adjusted[target_d] -= self.beta

        return reward_adjusted.view_as(reward_matrix)

    def calculate_reward(self, state_matrix):
        # SPGG 原始收益
        padded = F.pad(state_matrix.float().unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='circular')
        neighbor_coop = F.conv2d(padded, self.neibor_kernel).squeeze()

        c_single_profit = (self.r * neighbor_coop / 5) - 1
        d_single_profit = (self.r * neighbor_coop / 5)

        padded_c_profit = F.pad(c_single_profit.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='circular')
        padded_d_profit = F.pad(d_single_profit.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='circular')

        c_total_profit = F.conv2d(padded_c_profit, self.neibor_kernel).squeeze() + c_single_profit
        d_total_profit = F.conv2d(padded_d_profit, self.neibor_kernel).squeeze() + d_single_profit

        reward_matrix = torch.where(state_matrix.bool(), c_total_profit, d_total_profit)

        # FRAC-PPO 干预机制
        if self.zeta > 0 and self.beta > 0 and self.zeta < 1.0:
            reward_matrix = self._apply_quota_mechanism(state_matrix, reward_matrix)

        return reward_matrix

    def ppo_update(self):
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        rewards = torch.stack(self.rewards)
        next_states = torch.stack(self.next_states)
        dones = torch.stack(self.dones)

        with torch.no_grad():
            _, values = self.policy(states)
            _, next_values = self.policy(next_states)

        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            dones_float = dones[t].float()
            psi = rewards[t] + self.gamma * next_values[t] * (1 - dones_float) - values[t]
            advantages[t] = psi + self.gamma * self.gae_lambda * last_advantage
            last_advantage = advantages[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values

        for _ in range(self.ppo_epochs):
            for batch in self._make_batch(states, actions, old_log_probs, advantages, returns):
                state_b, action_b, old_log_b, adv_b, ret_b = batch
                if ret_b.shape[0] == 1:
                    ret_b = ret_b.squeeze()

                probs, value_pred = self.policy(state_b)
                dist = Categorical(probs)
                log_probs = dist.log_prob(action_b).view_as(action_b)
                entropy = dist.entropy().mean()

                ratio = (log_probs - old_log_b).exp()
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(value_pred, ret_b)

                loss = actor_loss + self.delta * critic_loss - self.rho * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                self.scheduler.step()

    def _make_batch(self, states, actions, old_log_probs, advantages, returns):
        perm = torch.randperm(len(states))
        for i in range(0, len(states), self.batch_size):
            idx = perm[i:i + self.batch_size]
            yield (states[idx], actions[idx], old_log_probs[idx], advantages[idx], returns[idx])

    async def run(self):
        coop_rates = []
        defect_rates = []
        total_values = []

        for epoch in tqdm(range(self.epochs)):
            self.epoch = epoch

            action, log_prob = self.choose_action(self.current_state)
            next_state = action
            reward = self.calculate_reward(next_state)
            done = torch.zeros_like(next_state, dtype=torch.bool)

            self.states.append(self.encode_state(self.current_state).view(self.L_num, self.L_num, 3))
            self.actions.append(action)
            self.log_probs.append(log_prob)

            self.rewards.append(reward)
            self.next_states.append(self.encode_state(next_state).view(self.L_num, self.L_num, 3))
            self.dones.append(done)

            if epoch == 0:
                profit_matrix = self.calculate_reward(self.current_state)
                task = asyncio.create_task(
                    self.shot_pic(self.current_state, epoch, self.r, profit_matrix)
                )
                self.shot_tasks.append(task)

            coop_rate = self.current_state.float().mean().item()
            defect_rate = 1 - coop_rate
            total_value = reward.sum().item()

            coop_rates.append(coop_rate)
            defect_rates.append(defect_rate)
            total_values.append(total_value)

            if len(self.states) >= self.batch_size * self.ppo_epochs:
                self.ppo_update()
                self.current_state = next_state
                self._reset_buffer()
            else:
                self.current_state = next_state

            if (epoch + 1) in [1, 10, 100, 1000, 10000, 100000]:
                profit_matrix = self.calculate_reward(self.current_state)
                task = asyncio.create_task(
                    self.shot_pic(self.current_state, epoch + 1, self.r, profit_matrix)
                )
                self.shot_tasks.append(task)

            if epoch % 1000 == 0:
                self.save_checkpoint()

            await asyncio.sleep(0)

        coop_rate = self.current_state.float().mean().item()
        defect_rate = 1 - coop_rate
        total_value = reward.sum().item()
        coop_rates.append(coop_rate)
        defect_rates.append(defect_rate)
        total_values.append(total_value)

        if self.shot_tasks:
            await asyncio.gather(*self.shot_tasks)

        self.save_checkpoint(is_final=True)
        return defect_rates, coop_rates, [], [], total_values

    def save_data(self, data_type, name, r, data):
        """保存实验数据：Density_C/D, Value_C/D, Total_Value"""
        output_dir = f'{self.output_path}/{data_type}'
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(f'{output_dir}/{name}.txt', data)

    def choose_action(self, state_matrix):
        with torch.no_grad():
            features = self.encode_state(state_matrix)
            probs, _ = self.policy(features)
            dist = Categorical(probs)
            actions = dist.sample()
            return actions.view_as(state_matrix), dist.log_prob(actions).view_as(state_matrix)

    def _reset_buffer(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.dones[:]

    def save_checkpoint(self, is_final=False):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'r': self.r,
            'gamma': self.gamma,
            'clip_epsilon': self.clip_epsilon
        }

        model_dir = f"{self.output_path}/checkpoint"
        os.makedirs(model_dir, exist_ok=True)
        filename = f"model_r{self.r}_final.pth" if is_final else f"model_r{self.r}_epoch{self.epoch}.pth"
        torch.save(checkpoint, f"{model_dir}/{filename}")

    async def shot_pic(self, type_t_matrix, epoch, r, profit_data):
        """保存策略快照和收益热图"""
        plt.clf()
        plt.close('all')

        img_dir = f'{self.output_path}/shot_pic/r={r}/two_type'
        matrix_dir = f'{self.output_path}/shot_pic/r={r}/two_type/type_t_matrix'
        profit_dir = f'{self.output_path}/shot_pic/r={r}/two_type/profit_matrix'

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(matrix_dir, exist_ok=True)
        os.makedirs(profit_dir, exist_ok=True)

        # 1. 策略矩阵可视化
        fig1 = plt.figure(figsize=(8, 8))
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.axis('off')
        fig1.patch.set_edgecolor('black')
        fig1.patch.set_linewidth(2)

        color_map = {0: [0, 0, 0], 1: [1, 1, 1]}
        strategy_image = np.zeros((self.L_num, self.L_num, 3))
        for label, color in color_map.items():
            strategy_image[type_t_matrix.cpu().numpy() == label] = color

        ax1.imshow(strategy_image, interpolation='none')
        ax1.axis('off')
        for spine in ax1.spines.values():
            spine.set_linewidth(3)

        fig1.savefig(f'{img_dir}/t={epoch}.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig1)

        # 2. 收益热图
        if isinstance(profit_data, tuple):
            combined_reward, _, team_utility = profit_data
            profit_matrix = combined_reward
        else:
            profit_matrix = profit_data

        if not isinstance(profit_matrix, torch.Tensor):
            profit_matrix = torch.tensor(profit_matrix, device=self.device)
        profit_matrix = profit_matrix.cpu().numpy()

        fig2 = plt.figure(figsize=(8, 8))
        ax2 = fig2.add_subplot(1, 1, 1)

        vmin = 0
        vmax = np.ceil(np.maximum(5 * (self.r - 1), 4 * self.r))
        im = ax2.imshow(profit_matrix, vmin=vmin, vmax=vmax, cmap='viridis', interpolation='none')

        cbar2 = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=28)
        cbar2.set_ticks(np.ceil(np.linspace(vmin, vmax, 5)).astype(int))

        fig2.savefig(f'{img_dir}/profit_t={epoch}.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig2)