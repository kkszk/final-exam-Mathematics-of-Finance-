# final-exam-Mathematics-of-Finance-
Financial Mathematics: Theory and AI Application

项目概述:
本项目基于《Machine Learning in Finance》Ch.9-Ch.12，实现了金融场景下的强化学习系列模型，涵盖信用风险闭环、期权对冲、逆强化学习与感知 - 决策闭环四大核心任务，支持离线训练、策略评估与安全部署。

代码结构：
code/
├── ch9.py          # Ch.9 信用风险闭环MDP建模 + FQI离线强化学习
├── ch10.py         # Ch.10 QLBS期权对冲强化学习
├── ch11.py         # Ch.11 AIRL逆强化学习（从专家日志学习奖励函数）
├── ch12.py         # Ch.12 感知-决策闭环 + LLM融合建模
├── requirements.txt # 依赖环境配置
└── README.md       # 一键复现指南

环境配置：
# 1. 创建虚拟环境（可选）
conda create -n p311 python=3.11
conda activate p311

# 2. 安装依赖
pip install -r requirements.txt

一键复现：
python ch9.py            #生成信贷离线数据、训练 FQI 算法、OPE 评估（WIS/FQE）、安全部署测试
python ch10.py           #模拟标的价格路径、构建对冲数据集、FQI 训练、对冲效果评估
python ch11.py           #生成专家日志、训练 AIRL 模型（恢复奖励函数 + 优化策略）、绘制训练损失图
python ch12.py           #LLM 语义编码、信息瓶颈压缩、KL 正则化策略训练、漂移监控
