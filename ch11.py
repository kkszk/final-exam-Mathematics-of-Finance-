import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
# -----------------------------
# 1. Synthetic Log Data Format
# -----------------------------
# Each trajectory: [(s_t, a_t, s_{t+1}), ...]
# Here we mock logs to make the prototype self-contained.

STATE_DIM = 8     # e.g. customer features, risk, balance, etc.
ACTION_DIM = 4    # e.g. {no action, SMS, call, escalate}


def generate_mock_logs(n_traj=50, traj_len=20):
    logs = []
    for _ in range(n_traj):
        traj = []
        s = np.random.randn(STATE_DIM)
        for t in range(traj_len):
            a = np.random.randint(ACTION_DIM)
            s_next = s + 0.1 * np.random.randn(STATE_DIM)
            traj.append((s, a, s_next))
            s = s_next
        logs.append(traj)
    return logs

expert_logs = generate_mock_logs()

# Flatten transitions for sampling
expert_transitions = [step for traj in expert_logs for step in traj]


# -----------------------------
# 2. Policy Network (pi_theta)
# -----------------------------

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, s):
        logits = self.net(s)
        return Categorical(logits=logits)


# -----------------------------
# 3. Reward Network r_psi(s,a)
#    AIRL-style shaping
# -----------------------------

class RewardNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, s, a_onehot):
        x = torch.cat([s, a_onehot], dim=1)
        return self.net(x)


# -----------------------------
# 4. AIRL Discriminator
#    D(s,a,s') = exp(f(s,a)) / (exp(f(s,a)) + pi(a|s))
# -----------------------------

class AIRLDiscriminator(nn.Module):
    def __init__(self, reward_net, policy_net):
        super().__init__()
        self.reward_net = reward_net
        self.policy_net = policy_net

    def forward(self, s, a, a_onehot):
        # f(s,a)
        r = self.reward_net(s, a_onehot).squeeze()

        # log pi(a|s)
        dist = self.policy_net(s)
        log_pi = dist.log_prob(a)

        # AIRL discriminator output (logits)
        logits = r - log_pi
        return torch.sigmoid(logits), logits


# -----------------------------
# 5. Utilities
# -----------------------------

def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


def one_hot(actions, action_dim):
    return torch.eye(action_dim)[actions]


# -----------------------------
# 6. Initialize Models
# -----------------------------

policy = PolicyNet(STATE_DIM, ACTION_DIM)
reward_net = RewardNet(STATE_DIM, ACTION_DIM)
discriminator = AIRLDiscriminator(reward_net, policy)

policy_optim = optim.Adam(policy.parameters(), lr=3e-4)
reward_optim = optim.Adam(reward_net.parameters(), lr=3e-4)


# -----------------------------
# 7. Training Loop (AIRL)
# -----------------------------

BATCH_SIZE = 64
EPOCHS = 500
GAMMA = 0.99

for epoch in range(EPOCHS):
    # ----- Sample expert transitions -----
    idx = np.random.choice(len(expert_transitions), BATCH_SIZE)
    batch = [expert_transitions[i] for i in idx]

    s_exp = to_tensor([b[0] for b in batch])
    a_exp = torch.tensor([b[1] for b in batch], dtype=torch.long)
    a_exp_oh = one_hot(a_exp, ACTION_DIM)

    # ----- Sample policy transitions -----
    with torch.no_grad():
        dist = policy(s_exp)
        a_pol = dist.sample()
        a_pol_oh = one_hot(a_pol, ACTION_DIM)

    # ----- Update reward / discriminator -----
    D_exp, logits_exp = discriminator(s_exp, a_exp, a_exp_oh)
    D_pol, logits_pol = discriminator(s_exp, a_pol, a_pol_oh)

    # Binary cross-entropy loss
    reward_loss = -(
        torch.log(D_exp + 1e-8).mean() +
        torch.log(1 - D_pol + 1e-8).mean()
    )

    reward_optim.zero_grad()
    reward_loss.backward()
    reward_optim.step()

    # ----- Update policy (using recovered reward) -----
    dist = policy(s_exp)
    log_pi = dist.log_prob(a_pol)

    with torch.no_grad():
        r_hat = reward_net(s_exp, a_pol_oh).squeeze()

    policy_loss = -(log_pi * r_hat).mean()

    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Reward loss: {reward_loss.item():.4f} | Policy loss: {policy_loss.item():.4f}")


print("Training complete.")



policy = PolicyNet(STATE_DIM, ACTION_DIM)
reward_net = RewardNet(STATE_DIM, ACTION_DIM)
discriminator = AIRLDiscriminator(reward_net, policy)
policy_optim = optim.Adam(policy.parameters(), lr=3e-4)
reward_optim = optim.Adam(reward_net.parameters(), lr=3e-4)

reward_loss_history = []
policy_loss_history = []

for epoch in range(EPOCHS):
    idx = np.random.choice(len(expert_transitions), BATCH_SIZE)
    batch = [expert_transitions[i] for i in idx]
    s_exp = to_tensor([b[0] for b in batch])
    a_exp = torch.tensor([b[1] for b in batch], dtype=torch.long)
    a_exp_oh = one_hot(a_exp, ACTION_DIM)

    with torch.no_grad():
        dist = policy(s_exp)
        a_pol = dist.sample()
        a_pol_oh = one_hot(a_pol, ACTION_DIM)

    D_exp, logits_exp = discriminator(s_exp, a_exp, a_exp_oh)
    D_pol, logits_pol = discriminator(s_exp, a_pol, a_pol_oh)

    reward_loss = -(
        torch.log(D_exp + 1e-8).mean() +
        torch.log(1 - D_pol + 1e-8).mean()
    )

    reward_optim.zero_grad()
    reward_loss.backward()
    reward_optim.step()

    dist = policy(s_exp)
    log_pi = dist.log_prob(a_pol)
    with torch.no_grad():
        r_hat = reward_net(s_exp, a_pol_oh).squeeze()
    policy_loss = -(log_pi * r_hat).mean()

    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    reward_loss_history.append(reward_loss.item())
    policy_loss_history.append(policy_loss.item())
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Reward loss: {reward_loss.item():.4f} | Policy loss: {policy_loss.item():.4f}")

plt.figure(figsize=(10, 6))
plt.plot(reward_loss_history, label='Reward/Discriminator Loss')
plt.plot(policy_loss_history, label='Policy Loss')
plt.title('AIRL Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig('奖励损失与策略损失变化趋势.png')
plt.close()

