import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. MDP 环境建模：信用风险闭环 =====================
class CreditRiskMDP:
    def __init__(self, gamma=0.95, interest_rate=0.01):
        self.gamma = gamma  # 折扣因子
        self.interest_rate = interest_rate  # 月利率
        self.action_space = np.array([-0.2, -0.1, 0, 0.1, 0.2])  # 动作空间
        self.state_dim = 6  # 状态维度：[资产, 负债, 逾期次数, 当前额度, 剩余生命周期, 征信评分]

    def generate_offline_data(self, n_samples=10000):
        """生成离线日志数据（模拟信贷业务历史数据）"""
        np.random.seed(42)
        # 1. 状态S：[资产, 负债, 逾期次数, 当前额度, 剩余生命周期, 征信评分]
        assets = np.random.normal(100, 30, n_samples).clip(min=10)  # 资产（万元）
        liabilities = np.random.normal(50, 20, n_samples).clip(min=0)  # 负债（万元）
        overdue_times = np.random.randint(0, 6, n_samples)  # 逾期次数（0-5）
        current_limit = np.random.normal(30, 10, n_samples).clip(min=5)  # 当前额度（万元）
        remaining_life = np.random.randint(1, 13, n_samples)  # 剩余生命周期（1-12月）
        credit_score = np.random.normal(70, 15, n_samples).clip(0, 100)  # 征信评分（0-100）
        S = np.column_stack([assets, liabilities, overdue_times, current_limit, remaining_life, credit_score])

        # 2. 行为策略μ(a|s)：历史动作分布（偏保守，上调动作少）
        def behavior_policy(s):
            """历史策略：征信评分<60则仅下调/维持，≥80则可上调"""
            cr = s[5]
            if cr < 60:
                return np.array([0.4, 0.4, 0.2, 0.0, 0.0])  # 仅-20%,-10%,0
            elif cr < 80:
                return np.array([0.2, 0.2, 0.4, 0.1, 0.1])  # 少量上调
            else:
                return np.array([0.1, 0.1, 0.3, 0.3, 0.2])  # 较多上调
        # 采样历史动作
        A = []
        for s in S:
            probs = behavior_policy(s)
            A.append(np.random.choice(self.action_space, p=probs))
        A = np.array(A)

        # 3. 奖励r(s,a)：利息收益 - 坏账损失
        bad_debt_prob = (liabilities / (assets + 1e-6)) * (overdue_times + 1) * 0.1  # 坏账概率
        interest_income = current_limit * (1 + A) * self.interest_rate  # 利息收益
        bad_debt_loss = current_limit * (1 + A) * bad_debt_prob  # 坏账损失
        R = interest_income - bad_debt_loss  # 奖励

        # 4. 下一状态S'
        assets_next = assets * (1 + np.random.normal(0, 0.02, n_samples))  # 资产小幅波动
        liabilities_next = liabilities * (1 + np.random.normal(0, 0.02, n_samples))  # 负债小幅波动
        # 逾期次数转移：上调额度+坏账概率高→逾期+1；下调额度+坏账概率低→逾期-1
        overdue_next = overdue_times + np.where(
            (A > 0) & (bad_debt_prob > 0.3), 1,
            np.where((A < 0) & (bad_debt_prob < 0.1), -1, 0)
        ).clip(min=0)
        current_limit_next = current_limit * (1 + A)  # 调整后额度
        remaining_life_next = remaining_life - 1  # 生命周期-1
        credit_score_next = credit_score + np.where(
            overdue_next > overdue_times, -5,  # 逾期增加→征信扣分
            np.where(overdue_next < overdue_times, +5, 0)  # 逾期减少→征信加分
        ).clip(0, 100)  # 征信评分边界
        S_next = np.column_stack([
            assets_next, liabilities_next, overdue_next,
            current_limit_next, remaining_life_next, credit_score_next
        ])

        # 整理数据
        data = pd.DataFrame({
            "S_asset": assets, "S_liability": liabilities, "S_overdue": overdue_times,
            "S_limit": current_limit, "S_life": remaining_life, "S_cr": credit_score,
            "A": A, "R": R,
            "S_next_asset": assets_next, "S_next_liability": liabilities_next,
            "S_next_overdue": overdue_next, "S_next_limit": current_limit_next,
            "S_next_life": remaining_life_next, "S_next_cr": credit_score_next
        })
        # 记录行为策略的动作概率（用于IS/WIS）
        data["mu_prob"] = [behavior_policy(s)[np.where(self.action_space == a)[0][0]] for s, a in zip(S, A)]
        return data

    def target_policy(self, s, q_model):
        """目标策略（FQI输出的最优策略）：argmax_a Q(s,a)"""
        q_values = []
        for a in self.action_space:
            feat = self._prepare_features(s.reshape(1, -1), np.array([a]))
            q_values.append(q_model.predict(feat)[0])
        return self.action_space[np.argmax(q_values)]

    def _prepare_features(self, S, A):
        """拼接状态和动作作为特征"""
        if len(S.shape) == 1:
            S = S.reshape(1, -1)
        A = A.reshape(-1, 1)
        return np.hstack([S, A])
    

# ===================== 2. FQI 算法实现 =====================
class FittedQIteration:
    def __init__(self, action_space, gamma=0.95, n_iterations=10, model=RandomForestRegressor()):
        self.action_space = action_space
        self.gamma = gamma
        self.n_iterations = n_iterations
        self.model = model
        self.q_models = []  # 存储每轮迭代的Q模型

    def fit(self, data):
        """训练FQI"""
        # 提取数据
        S_cols = ["S_asset", "S_liability", "S_overdue", "S_limit", "S_life", "S_cr"]
        S_next_cols = ["S_next_asset", "S_next_liability", "S_next_overdue", "S_next_limit", "S_next_life", "S_next_cr"]
        S = data[S_cols].values
        A = data["A"].values
        R = data["R"].values
        S_next = data[S_next_cols].values

        # 初始化Q0 = R
        X = self._prepare_features(S, A)
        self.model.fit(X, R)
        self.q_models.append(self.model)

        # 迭代拟合Q函数
        for i in range(1, self.n_iterations):
            # 计算V_next = max_a Q_prev(S_next, a)
            V_next = []
            for s_next in S_next:
                q_values = []
                for a in self.action_space:
                    feat = self._prepare_features(s_next.reshape(1, -1), np.array([a]))
                    q_values.append(self.q_models[-1].predict(feat)[0])
                V_next.append(max(q_values))
            V_next = np.array(V_next)

            # 目标Q值：Q = R + γ * V_next
            Q_target = R + self.gamma * V_next

            # 拟合新的Q模型
            self.model = self.model.__class__(**self.model.get_params())  # 重置模型
            self.model.fit(X, Q_target)
            self.q_models.append(self.model)
            print(f"FQI Iteration {i} | MSE: {np.mean((self.model.predict(X) - Q_target)**2):.4f}")

    def _prepare_features(self, S, A):
        if len(S.shape) == 1:
            S = S.reshape(1, -1)
        A = A.reshape(-1, 1)
        return np.hstack([S, A])
    
# ===================== 3. Offline Policy Evaluation (OPE) =====================
class OfflinePolicyEvaluation:
    def __init__(self, mdp, fqi_model):
        self.mdp = mdp
        self.fqi_model = fqi_model
        self.action_space = mdp.action_space

    def weighted_importance_sampling(self, data, n_bootstrap=1000):
        """加权重要性采样（WIS）+ Bootstrap不确定性"""
        # 提取数据
        S_cols = ["S_asset", "S_liability", "S_overdue", "S_limit", "S_life", "S_cr"]
        S = data[S_cols].values
        A = data["A"].values
        R = data["R"].values
        mu_probs = data["mu_prob"].values

        # 目标策略π(a|s)的概率
        pi_probs = []
        for s, a in zip(S, A):
            # 计算目标策略对当前动作a的概率（确定性策略：选最优动作则概率=1，否则=0）
            optimal_a = self.mdp.target_policy(s, self.fqi_model.q_models[-1])
            pi_probs.append(1.0 if a == optimal_a else 0.0)
        pi_probs = np.array(pi_probs)

        # 计算重要性权重
        weights = pi_probs / (mu_probs + 1e-6)
        weights = weights / np.sum(weights)  # 归一化（WIS）

        # 计算基线WIS估值
        wis_estimate = np.sum(weights * R)

        # Bootstrap 计算不确定性
        bootstrap_estimates = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(data), size=len(data), replace=True)
            w_boot = weights[idx]
            w_boot = w_boot / np.sum(w_boot)
            r_boot = R[idx]
            bootstrap_estimates.append(np.sum(w_boot * r_boot))

        # 输出结果
        mean_est = np.mean(bootstrap_estimates)
        ci_95 = (np.percentile(bootstrap_estimates, 2.5), np.percentile(bootstrap_estimates, 97.5))
        return {
            "WIS_base": wis_estimate,
            "WIS_bootstrap_mean": mean_est,
            "WIS_95_CI": ci_95
        }

    def fitted_q_evaluation(self, data, n_sensitivity=5):
        """拟合Q评估（FQE）+ 敏感性分析"""
        # 提取状态分布
        S_cols = ["S_asset", "S_liability", "S_overdue", "S_limit", "S_life", "S_cr"]
        S = data[S_cols].values

        # 基线FQE估值：E[max_a Q(s,a)]
        fqe_estimates = []
        for s in S:
            q_values = []
            for a in self.action_space:
                feat = self.mdp._prepare_features(s.reshape(1, -1), np.array([a]))
                q_values.append(self.fqi_model.q_models[-1].predict(feat)[0])
            fqe_estimates.append(max(q_values))
        base_est = np.mean(fqe_estimates)

        # 敏感性分析：调整FQI迭代次数
        sensitivity_results = []
        for n_iter in [5, 8, 10, 12, 15]:
            # 重新训练FQI（不同迭代次数）
            fqi_temp = FittedQIteration(
                action_space=self.action_space,
                gamma=self.mdp.gamma,
                n_iterations=n_iter,
                model=RandomForestRegressor(n_estimators=50, random_state=42)
            )
            fqi_temp.fit(data)
            # 重新计算FQE
            temp_est = []
            for s in S:
                q_values = []
                for a in self.action_space:
                    feat = self.mdp._prepare_features(s.reshape(1, -1), np.array([a]))
                    q_values.append(fqi_temp.q_models[-1].predict(feat)[0])
                temp_est.append(max(q_values))
            sensitivity_results.append({
                "n_iter": n_iter,
                "fqe_estimate": np.mean(temp_est)
            })

        return {
            "FQE_base": base_est,
            "sensitivity_analysis": sensitivity_results
        }

# ===================== 4. 上线安全/合规工具 =====================
class DeploymentSafety:
    def __init__(self, mdp):
        self.mdp = mdp
        self.psi_threshold = 0.2  # PSI阈值（>0.2视为分布漂移）
        self.bad_debt_threshold = 0.03  # 坏账率阈值（>3%触发熔断）

    def psi_calculation(self, ref_dist, curr_dist):
        """计算PSI（Population Stability Index）检测分布漂移"""
        # 分箱（10个等频箱）
        ref_bins = pd.qcut(ref_dist, 10, duplicates="drop")
        curr_bins = pd.cut(curr_dist, bins=ref_bins.cat.categories)
        # 计算每个箱的占比
        ref_counts = ref_bins.value_counts(normalize=True)
        curr_counts = curr_bins.value_counts(normalize=True)
        # 对齐索引
        all_bins = ref_counts.index.union(curr_counts.index)
        ref_counts = ref_counts.reindex(all_bins, fill_value=0)
        curr_counts = curr_counts.reindex(all_bins, fill_value=0)
        # 计算PSI
        psi = np.sum((curr_counts - ref_counts) * np.log((curr_counts + 1e-6) / (ref_counts + 1e-6)))
        return psi

    def distribution_shift_detection(self, ref_data, curr_data):
        """检测状态/奖励分布漂移"""
        shift_results = {}
        # 检测资产特征漂移
        ref_asset = ref_data["S_asset"].values
        curr_asset = curr_data["S_asset"].values
        shift_results["S_asset_PSI"] = self.psi_calculation(ref_asset, curr_asset)
        shift_results["S_asset_KS"] = ks_2samp(ref_asset, curr_asset).pvalue
        # 检测奖励漂移
        ref_r = ref_data["R"].values
        curr_r = curr_data["R"].values
        shift_results["R_PSI"] = self.psi_calculation(ref_r, curr_r)
        shift_results["R_KS"] = ks_2samp(ref_r, curr_r).pvalue
        # 检测动作分布漂移
        ref_a = ref_data["A"].values
        curr_a = curr_data["A"].values
        shift_results["A_PSI"] = self.psi_calculation(ref_a, curr_a)
        shift_results["A_KS"] = ks_2samp(ref_a, curr_a).pvalue
        # 判断是否漂移
        shift_results["is_shift"] = any([
            shift_results["S_asset_PSI"] > self.psi_threshold,
            shift_results["R_PSI"] > self.psi_threshold,
            shift_results["A_PSI"] > self.psi_threshold
        ])
        return shift_results

    def risk_guardrail(self, s):
        """风险护栏：限制高风险客户的动作空间"""
        cr = s[5]  # 征信评分
        overdue = s[2]  # 逾期次数
        # 高风险客户：征信<60 或 逾期≥3次
        if cr < 60 or overdue >= 3:
            return np.array([-0.2, -0.1, 0])  # 禁止上调额度
        else:
            return self.mdp.action_space  # 允许所有动作

    def audit_logger(self, s, a, r, s_next):
        """审计日志：记录全链路数据"""
        log = {
            "timestamp": pd.Timestamp.now(),
            "state": s.tolist(),
            "action": a,
            "reward": r,
            "next_state": s_next.tolist(),
            "risk_guardrail_applied": a not in self.mdp.action_space  # 是否触发护栏
        }
        return log
    
# ===================== 5. 主流程运行 =====================
if __name__ == "__main__":
    # 1. 初始化MDP环境
    mdp = CreditRiskMDP(gamma=0.95)
    # 生成离线数据
    offline_data = mdp.generate_offline_data(n_samples=10000)
    print("离线数据示例：\n", offline_data.head())

    # 2. 训练FQI算法
    fqi = FittedQIteration(
        action_space=mdp.action_space,
        gamma=0.95,
        n_iterations=10,
        model=RandomForestRegressor(n_estimators=50, random_state=42)
    )
    print("\n开始训练FQI：")
    fqi.fit(offline_data)

    # 3. 离线策略评估（OPE）
    ope = OfflinePolicyEvaluation(mdp, fqi)
    # 3.1 WIS评估
    wis_results = ope.weighted_importance_sampling(offline_data)
    print("\nWIS评估结果：")
    print(f"基线估值：{wis_results['WIS_base']:.4f}")
    print(f"Bootstrap均值：{wis_results['WIS_bootstrap_mean']:.4f}")
    print(f"95%置信区间：{wis_results['WIS_95_CI']}")

    # 3.2 FQE评估
    fqe_results = ope.fitted_q_evaluation(offline_data)
    print("\nFQE评估结果：")
    print(f"基线估值：{fqe_results['FQE_base']:.4f}")
    print("敏感性分析（不同迭代次数）：")
    for res in fqe_results["sensitivity_analysis"]:
        print(f"迭代次数{res['n_iter']} → FQE估值：{res['fqe_estimate']:.4f}")

    # 4. 上线安全/合规测试
    safety = DeploymentSafety(mdp)
    # 4.1 分布漂移检测（模拟线上数据）
    online_data = mdp.generate_offline_data(n_samples=2000)  # 模拟线上数据
    shift_results = safety.distribution_shift_detection(offline_data, online_data)
    print("\n分布漂移检测结果：")
    print(f"资产PSI：{shift_results['S_asset_PSI']:.4f} (阈值={safety.psi_threshold})")
    print(f"奖励PSI：{shift_results['R_PSI']:.4f}")
    print(f"是否漂移：{shift_results['is_shift']}")

    # 4.2 风险护栏测试
    high_risk_s = np.array([80, 90, 4, 20, 6, 55])  # 高风险客户：负债高、逾期4次、征信55
    allowed_actions = safety.risk_guardrail(high_risk_s)
    print("\n高风险客户允许的动作：", allowed_actions)

    # 4.3 审计日志示例
    s = np.array([120, 30, 0, 40, 6, 85])  # 低风险客户
    a = mdp.target_policy(s, fqi.q_models[-1])  # 最优动作
    log = safety.audit_logger(s, a, 0.4, np.array([122, 29, 0, 44, 5, 88]))
    print("\n审计日志示例：")
    print(log)