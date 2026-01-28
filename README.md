# Meta Reinforcement Learning for Battery Charging

该代码库实现了一个将 DDPG 强化学习代理与 PyBaMM 单粒子电池模型 (SPMe) 结合的元学习框架。目标是在不同电池参数集上学习高效、安全的充电策略，并支持元策略迁移、离线复现以及可视化分析。

## 仓库结构
- `RL_train.py`：包含训练/评估流程、Reptile 元更新、绘图与辅助函数。
- `SPM.py`：封装 PyBaMM SPMe 仿真环境，实现 `step`、`reset` 等交互接口。
- `ddpg.py`：DDPG 代理、噪声与经验回放缓冲区。
- `model.py`：Actor/Critic 网络结构定义。

## 环境准备
1. **创建虚拟环境（建议 Python ≥3.9）**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
   ```
2. **安装依赖**
   ```bash
   pip install --upgrade pip
   pip install torch numpy matplotlib ipdb pybamm
   ```
   - PyBaMM 在首次运行时会尝试下载求解器，如受网络限制需要提前手动安装 Sundials/CasADi，请参考 PyBaMM 官方文档。

## 快速上手
以下示例展示了从零开始训练、评估与保存策略的最小流程。所有示例均可在仓库根目录的 Python REPL 或脚本中运行。

### 1. 初始化代理
```python
from ddpg import Agent

agent = Agent(state_size=3, action_size=1, random_seed=0)
```

### 2. 单任务 (DDPG) 训练
`inner_ddpg` 会在指定的电池参数集上运行 `m` 个 episode，并使用经验回放更新 agent。

```python
from RL_train import inner_ddpg

scores = inner_ddpg(agent, n_episodes=10, param="Chen2020")
```

### 3. Reptile 元训练
使用多个 PyBaMM 参数集（即不同的电池）交替训练，提升策略的泛化能力。

```python
import numpy as np
from RL_train import reptile

param_sets = ["Chen2020", "Mohtat2020", "Ai2020", "Xu2020"]  # PyBaMM 内置参数名称
meta_lr = 0.05            # 元更新步长
inner_episodes = 5        # 每个任务的 inner-loop 长度

for iteration in range(50):
    scores = reptile(agent, param_sets, m=inner_episodes, meta_lr=meta_lr)
    print(f"Iter {iteration:03d} | mean reward {np.mean(scores):.2f}")
```

### 4. 策略评估
`eval_policy` 会返回整个充电序列中的电流轨迹、平均奖励、温度/电压峰值及充电步数。

```python
from RL_train import eval_policy

currents, avg_r, max_temp, max_volt, steps = eval_policy(agent, eval_episodes=3, param="Chen2020")
```

### 5. 保存 / 加载参数
```python
from RL_train import save_agent_parameters, load_agent_parameters

save_agent_parameters(agent, "checkpoints/meta_agent.pt")

new_agent = Agent(state_size=3, action_size=1, random_seed=0)
load_agent_parameters(new_agent, "checkpoints/meta_agent.pt")
```

## 功能详解
### 环境 (`SPM.py`)
- 默认采样时间 `sample_time=90s`，动作单位为安培 (A)，正值代表充电电流。
- `param` 参数可选用 PyBaMM 自带的电池参数化字符串（如 `"Chen2020"`），也可自行传入 `pybamm.ParameterValues` 对象。
- 约束由 `constraints temperature max` 和 `constraints voltage max` 控制，可根据需求修改。

### 训练脚本 (`RL_train.py`)
- `normalize_outputs` / `denormalize_input`：在网络和物理量之间做尺度变换，方便稳定训练。
- `inner_ddpg`：单任务 DDPG 训练循环，内部调用 `Agent.step` 完成经验回放与采样。
- `reptile`：实现 Reptile 元学习算法，随机抽取任务、运行 `inner_ddpg`，再用 `meta_lr` 做一次全局参数更新。
- `plot_total_rewards`、`plot_battery_state`：可视化训练奖励或复现某一充电电流序列下的电压/温度/SOC 曲线。
- `charge_battery_with_sequence`：如需复现外部政策，可直接传入一段电流序列并查看仿真结果。

## 可视化与结果分析
1. **奖励曲线**
   ```python
   from RL_train import plot_total_rewards
   plot_total_rewards(scores)
   ```
2. **充电过程**
   ```python
   from RL_train import charge_battery_with_sequence, plot_battery_state

   test_currents = [3.0] * 50  # 50 个 90s 步长
   v, t, soc = charge_battery_with_sequence(test_currents, param="Chen2020")
   plot_battery_state(v, t, soc, test_currents)
   ```

## 典型工作流
1. 选择或编写适合的电池参数集列表。
2. 调整 `inner_ddpg` episode 数、Reptile 的 `meta_lr` 与 `m` 等超参，执行若干迭代训练。
3. 使用 `eval_policy` 检查策略是否满足温度/电压约束并达到目标 SOC。
4. 利用 `charge_battery_with_sequence` 和绘图函数对策略输出的电流序列进行物理可解释性验证。
5. 保存模型权重，供后续部署或迁移学习使用。

## 常见问题
- **PyBaMM 报错缺少求解器**：需要按照 PyBaMM 官方指南安装 Sundials（或 CasADi）。macOS/Linux 可通过 `brew install sundials` 或 `conda install -c conda-forge sundials`。
- **求解速度慢**：可减少 `inner_ddpg` 的 episode，或在 `SPM.sett['sample_time']` 中增大步长以降低求解频率，但需权衡精度。
- **动作越界**：`Agent.act` 输出已被裁剪到 `[-1, 1]`，通过 `denormalize_input` 映射到 `1~7.5A`。若需更大电流，请同步调整该函数。

## 参考
- [PyBaMM documentation](https://pybamm.org/)
- OpenAI DDPG baseline 和 Reptile 元学习框架

如需扩展到其他强化学习算法，可复用 `SPM` 环境接口，并根据需求调整 reward 或约束定义。
