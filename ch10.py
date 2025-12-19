import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# =============================
# 1. 市场与期权参数
# =============================

S0 = 100.0        # 初始标的价格
K = 100.0         # 行权价
T = 1.0           # 到期时间（年）
N = 50            # 离散时间步数
DT = T / N
MU = 0.0          # 漂移（风险中性下可设为 0）
SIGMA = 0.2       # 波动率
R = 0.0           # 无风险利率

# 交易与风险参数
LAMBDA_RISK = 1e-2     # 风险厌恶系数（方差惩罚）
COST = 1e-3            # 交易成本系数
POSITION_LIMIT = 5.0  # 最大持仓限制


# =============================
# 2. MDP 定义（QLBS）
# =============================
# 状态 s_t = (S_t, t, h_t)
#   S_t : 标的价格
#   t   : 时间索引
#   h_t : 当前持仓（delta）
# 动作 a_t = Δh_t（调仓量）
# 奖励 r_t = - 对冲误差^2 - 风险惩罚 - 交易成本


def simulate_price_paths(n_paths=2000):
    """
    使用几何布朗运动模拟标的价格路径
    """
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = S0
    for t in range(N):
        z = np.random.randn(n_paths)
        paths[:, t+1] = paths[:, t] * np.exp(
            (MU - 0.5 * SIGMA**2) * DT + SIGMA * np.sqrt(DT) * z
        )
    return paths


# =============================
# 3. 奖励函数（对冲 PnL）
# =============================

def option_payoff(S):
    """
    欧式看涨期权到期收益
    """
    return np.maximum(S - K, 0.0)


def reward_fn(S_t, S_tp1, h_t, a_t):
    """
    单步奖励：
    - PnL 误差平方
    - 风险惩罚（持仓方差）
    - 交易成本
    """
    h_tp1 = np.clip(h_t + a_t, -POSITION_LIMIT, POSITION_LIMIT)

    # 对冲组合的变化
    pnl = h_t * (S_tp1 - S_t)

    # 风险惩罚 + 交易成本
    reward = - pnl**2 \
             - LAMBDA_RISK * h_tp1**2 \
             - COST * np.abs(a_t)
    return reward, h_tp1


# =============================
# 4. 数据集构造（FQI 批量样本）
# =============================
# 样本格式：(X_t, a_t, r_t, X_{t+1})


def build_dataset(price_paths, action_grid):
    dataset = []
    for path in price_paths:
        h = 0.0  # 初始持仓
        for t in range(N):
            S_t = path[t]
            S_tp1 = path[t+1]
            for a in action_grid:
                r, h_next = reward_fn(S_t, S_tp1, h, a)
                state = np.array([S_t, t, h])
                next_state = np.array([S_tp1, t+1, h_next])
                dataset.append((state, a, r, next_state))
    return dataset


# =============================
# 5. Fitted Q-Iteration (FQI)
# =============================

class QRegressor:
    """
    使用多项式特征 + Ridge 回归逼近 Q(s,a)
    """
    def __init__(self, degree=2, alpha=1e-4):
        self.poly = PolynomialFeatures(degree)
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y):
        Xp = self.poly.fit_transform(X)
        self.model.fit(Xp, y)

    def predict(self, X):
        Xp = self.poly.transform(X)
        return self.model.predict(Xp)


# =============================
# 6. FQI 主训练流程
# =============================

price_paths = simulate_price_paths()
action_grid = np.linspace(-1.0, 1.0, 5)  # 离散调仓动作

dataset = build_dataset(price_paths, action_grid)

# 划分训练 / 验证集
np.random.shuffle(dataset)
split = int(0.8 * len(dataset))
train_data = dataset[:split]
val_data = dataset[split:]

q_model = QRegressor(degree=2)
GAMMA = 1.0  # QLBS 中常用无折现（终端 payoff 已含时间）

for it in range(10):
    X, y = [], []
    for (s, a, r, s_next) in train_data:
        # 构造回归输入
        X.append(np.concatenate([s, [a]]))

        # Bellman target
        q_next = []
        for a_next in action_grid:
            q_next.append(
                q_model.predict([
                    np.concatenate([s_next, [a_next]])
                ])[0]
            )
        target = r + GAMMA * np.max(q_next)
        y.append(target)

    q_model.fit(np.array(X), np.array(y))
    print(f"FQI iteration {it} finished")


