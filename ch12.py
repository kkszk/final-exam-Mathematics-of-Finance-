import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib as plt
# -----------------------------
# 1. 环境与观测定义
# -----------------------------

# ot：原始观测，这里用文本 embedding 代替真实 LLM 输出
OBS_DIM = 16     # LLM 输出的语义向量维度（示意）
ACTION_DIM = 4   # 动作空间：{安抚, 催收, 升级人工, 结束}


def mock_llm_encoder(text_batch):
    """
    模拟 LLM 感知模块
    输入：文本（字符串列表）
    输出：语义表示 z_t ~ f_ϕ(o_t)
    实际系统中此处可接 GPT / Claude / 本地大模型
    """
    batch_size = len(text_batch)
    return torch.randn(batch_size, OBS_DIM)


# -----------------------------
# 2. 表示层（Information Bottleneck）
# -----------------------------

class BottleneckEncoder(nn.Module):
    """
    信息瓶颈层：
    限制 z_t 中的信息量，防止过拟合具体措辞
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, latent_dim)

    def forward(self, z):
        return self.fc(z)


# -----------------------------
# 3. 策略网络 π(a|z)
# -----------------------------

class PolicyNet(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, z):
        logits = self.net(z)
        return Categorical(logits=logits)


# -----------------------------
# 4. KL 正则：受限理性策略
# -----------------------------

class PriorPolicy(nn.Module):
    """
    参考策略（例如规则系统 / 保守策略）
    用于 KL 正则，限制新策略偏离
    """
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.net = nn.Linear(latent_dim, action_dim)

    def forward(self, z):
        return Categorical(logits=self.net(z))


# -----------------------------
# 5. 初始化模块
# -----------------------------

encoder = BottleneckEncoder(OBS_DIM, latent_dim=8)
policy = PolicyNet(latent_dim=8, action_dim=ACTION_DIM)
prior_policy = PriorPolicy(latent_dim=8, action_dim=ACTION_DIM)

optimizer = optim.Adam(list(encoder.parameters()) + list(policy.parameters()), lr=3e-4)

BETA = 0.1  # KL 正则强度（bounded rationality）


# -----------------------------
# 6. 训练 / 决策循环（示意）
# -----------------------------

for step in range(1000):
    # ---- 感知阶段：LLM 编码 ----
    texts = ["我最近资金有点紧", "你们催得太频繁了"]
    z_llm = mock_llm_encoder(texts)

    # ---- 信息瓶颈压缩 ----
    z = encoder(z_llm)

    # ---- 策略采样动作 ----
    dist = policy(z)
    actions = dist.sample()

    # ---- 模拟奖励（示意） ----
    # 实际中来自回款率、满意度、合规等
    reward = torch.randn(len(actions))

    # ---- KL 正则项 ----
    prior_dist = prior_policy(z)
    kl_div = torch.distributions.kl.kl_divergence(dist, prior_dist).mean()

    # ---- 总损失（端到端联合优化） ----
    loss = -(dist.log_prob(actions) * reward).mean() + BETA * kl_div

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step:03d} | Loss={loss.item():.4f} | KL={kl_div.item():.4f}")


import matplotlib.pyplot as plt
import numpy as np

# 补充训练过程中的数据记录（需要在训练循环中添加记录逻辑）
# 修改原训练循环，添加历史记录存储
loss_history = []
kl_history = []

# 重新运行训练循环并记录数据
encoder = BottleneckEncoder(OBS_DIM, latent_dim=8)
policy = PolicyNet(latent_dim=8, action_dim=ACTION_DIM)
prior_policy = PriorPolicy(latent_dim=8, action_dim=ACTION_DIM)
optimizer = optim.Adam(list(encoder.parameters()) + list(policy.parameters()), lr=3e-4)

for step in range(1000):
    texts = ["我最近资金有点紧", "你们催得太频繁了"]
    z_llm = mock_llm_encoder(texts)
    z = encoder(z_llm)
    dist = policy(z)
    actions = dist.sample()
    reward = torch.randn(len(actions))
    prior_dist = prior_policy(z)
    kl_div = torch.distributions.kl.kl_divergence(dist, prior_dist).mean()
    loss = -(dist.log_prob(actions) * reward).mean() + BETA * kl_div
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    kl_history.append(kl_div.item())
    
    if step % 20 == 0:
        print(f"Step {step:03d} | Loss={loss.item():.4f} | KL={kl_div.item():.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history, label='Training Loss')
plt.title('Loss Curve')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(kl_history, label='KL Divergence', color='orange')
plt.title('KL Divergence Curve')
plt.xlabel('Step')
plt.ylabel('KL Value')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(' Loss 与 KL 散度的联合变化趋势.png')
plt.close()
