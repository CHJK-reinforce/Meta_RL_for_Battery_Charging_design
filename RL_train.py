import torch
import ipdb
from ddpg import Agent
from SPM import SPM
import numpy as np
import time
import copy
import random
import matplotlib.pyplot as plt
import time

BUFFER_SIZE = int(1e5)     # 池子深度
BATCH_SIZE = 64            # 批次尺寸
action_size = 1
gamma = 0.99
TAU = 1e-3


# 正则化电池的状态量（由此可以看出SOC，电压，温度的范围）
def normalize_outputs(soc, voltage, temperature):
    norm_soc = (soc - 0.5)*2
    norm_voltage = (voltage - 3.5) / 1
    norm_temperature = (temperature - 308) / 11
    norm_output = np.array([norm_soc, norm_voltage, norm_temperature])

    return norm_output


def get_output_observations(bat):
    return bat.soc, bat.voltage, bat.temp


def denormalize_input(input_value, min_output_value, max_output_value):
    output_value = (1 + input_value) * (max_output_value - min_output_value) / 2 + min_output_value

    return output_value

def eval_policy(policy, eval_episodes=3, param = "Chen2020"):
    # 电池模型初始化
    start_time = time.time()
    eval_env = SPM(3.2, 303, param)

    # 初始化奖励，违反量，充电时间为0
    avg_reward = 0.
    avg_temp_vio = 0.
    avg_volt_vio = 0.
    avg_chg_time = 0.
    list_i = []
    r_plat_all = []

    for n in range(eval_episodes):
        # 初始化在范围内随机初始化
        # initial_conditions['init_v'] = np.random.uniform(low=2.8, high=3.2)
        initial_conditions['init_v'] = 3.2
        # initial_conditions['init_t'] = np.random.uniform(low=298, high=302)
        initial_conditions['init_t'] = 303

        # 按照给定条件设置模型
        state, done = eval_env.reset(init_v=initial_conditions['init_v'], init_t=initial_conditions['init_t']), False
        soc, voltage, temperature = get_output_observations(eval_env)
        # norm_out是正则化后的状态量
        norm_out = normalize_outputs(soc, voltage, temperature)

        action_vec = []
        t_vec = []
        v_vec = []
        score = 0
        while not done:
            # 从策略policy中得到下一步动作
            norm_action = policy.act(norm_out, add_noise=False)
            # 去正则化，实际的动作
            applied_action = denormalize_input(norm_action, 1, 7.5)
            list_i.append(applied_action)

            # 电池模型，启动！
            _, reward, done, _ = eval_env.step(applied_action)

            next_soc, next_voltage, next_temperature = get_output_observations(eval_env)
            norm_next_out = normalize_outputs(next_soc, next_voltage, next_temperature)

            # vec应该是记录的条带
            action_vec.append(applied_action)
            t_vec.append(eval_env.temp)
            v_vec.append(eval_env.voltage)
            score += reward

            norm_out = norm_next_out

            avg_reward += reward

        avg_temp_vio += np.max(np.array(t_vec))
        avg_volt_vio += np.max(np.array(v_vec))
        avg_chg_time += len(action_vec)
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")

    avg_reward /= eval_episodes
    avg_temp_vio /= eval_episodes
    avg_volt_vio /= eval_episodes
    avg_chg_time /= eval_episodes

    avg_max_volt_vio = avg_volt_vio
    avg_max_temp_vio = avg_temp_vio

    end_time = time.time()


    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} time:{end_time-start_time}")
    print("---------------------------------------")

    return list_i, avg_reward, avg_max_temp_vio, avg_max_volt_vio, avg_chg_time

def compute_statistics(values):
    """
    计算给定值的最大值、最小值和平均值
    :param values: 列表或数组
    :return: 最大值, 最小值, 平均值
    """
    max_value = np.max(values)
    min_value = np.min(values)
    avg_value = np.mean(values)

    return max_value, min_value, avg_value
def inner_ddpg(agent, n_episodes=10, param="Chen2020"):
    checkpoint_list = []
    scores_list = []

    # 保存网络的初始参数
    i_episode = 0
    checkpoint_list.append(i_episode)
    env = SPM(3.2, 303, param)

    for i_episode in range(1, n_episodes + 1):
        # 随机的初始值
        # initial_conditions['init_v'] = np.random.uniform(low=2.8, high=3.2)
        initial_conditions['init_v'] = 3.2
        # initial_conditions['init_t'] = np.random.uniform(low=298, high=302)
        initial_conditions['init_t'] = 303

        # 初始化电池得到现在的状态量
        _ = env.reset(init_v=initial_conditions['init_v'], init_t=initial_conditions['init_t'])
        soc, voltage, temperature = get_output_observations(env)
        norm_out = normalize_outputs(soc, voltage, temperature)

        # noise的重置
        agent.reset()

        score = 0
        done = False
        cont = 0

        while not done or score > -1000:

            norm_action = agent.act(norm_out, add_noise=True)

            # 实际的动作
            applied_action = denormalize_input(norm_action, 1, 7.5)

            # 执行动作
            # reward 由环境自动给出
            _, reward, done, _ = env.step(applied_action)
            next_soc, next_voltage, next_temperature = get_output_observations(env)
            norm_next_out = normalize_outputs(next_soc, next_voltage, next_temperature)

            # 这一步更新网络，意思是一次完整的充电后更新一次网络
            agent.step(norm_out, norm_action, reward, norm_next_out, done)
            try:
                score += reward
            except:
                ipdb.set_trace()
            cont += 1
            if done:
                break

            norm_out = norm_next_out
        print(score)

        scores_list.append(score)

    return scores_list

def charge_battery_with_sequence(current_sequence, param="Chen2020"):
    env = SPM(3.2, 303, param)
    # initial_conditions['init_v'] = np.random.uniform(low=2.8, high=3.2)
    initial_conditions['init_v'] = 3.2
    # initial_conditions['init_t'] = np.random.uniform(low=298, high=302)
    initial_conditions['init_t'] = 303
    env.reset(init_v=initial_conditions['init_v'], init_t=initial_conditions['init_t']), False

    voltages = []
    temperatures = []
    socs = []

    for current in current_sequence:
        _, reward, done, _ = env.step(current)
        soc, voltage, temperature = get_output_observations(env)
        voltages.append(voltage)
        temperatures.append(temperature)
        socs.append(soc)
        if done:
            break

    return voltages, temperatures, socs

def plot_battery_state(voltages, temperatures, socs, current_sequence):
    time_steps = [i * 90 for i in range(len(voltages))]

    plt.figure(figsize=(11, 8))

    plt.subplot(4, 1, 1)
    plt.plot(time_steps, voltages, color='blue', linewidth=3)
    plt.axhline(y=4.2, color='red', linestyle='--', linewidth=4)
    plt.title("Voltage(V)", fontweight="bold", fontsize=28)
    # plt.ylabel("Voltage(V)", fontweight="bold", fontsize=17)
    plt.legend(fontsize=20, loc='upper left', fancybox=True, framealpha=0.5,
           prop={'weight':'bold'})

    ax = plt.gca()
    ax.spines['top'].set_linewidth(3)  # 调整上边框线宽
    ax.spines['right'].set_linewidth(3)  # 调整右边框线宽
    ax.spines['left'].set_linewidth(3)  # 调整左边框线宽
    ax.spines['bottom'].set_linewidth(3)

    plt.xticks(fontsize=26, fontweight='bold')
    plt.yticks(fontsize=28, fontweight='bold')

    plt.subplot(4, 1, 2)
    plt.plot(time_steps, temperatures, color='red', linewidth=3)
    plt.axhline(y=309, color='b', linestyle='--', linewidth=4)
    plt.title("Temperature(K)", fontweight="bold", fontsize=28)
    # plt.ylabel("Temperature(K)", fontweight="bold", fontsize=17)
    plt.legend(fontsize=20, loc='upper left', fancybox=True, framealpha=0.5,
           prop={'weight':'bold'})

    ax = plt.gca()
    ax.spines['top'].set_linewidth(3)  # 调整上边框线宽
    ax.spines['right'].set_linewidth(3)  # 调整右边框线宽
    ax.spines['left'].set_linewidth(3)  # 调整左边框线宽
    ax.spines['bottom'].set_linewidth(3)

    plt.xticks(fontsize=26, fontweight='bold')
    plt.yticks(fontsize=24, fontweight='bold')

    plt.subplot(4, 1, 3)
    plt.plot(time_steps, socs, color='green', linewidth=3)
    plt.title("SOC", fontweight="bold", fontsize=28)
    # plt.ylabel("SOC", fontweight="bold", fontsize=18)
    plt.legend(fontsize=35, loc='upper right', fancybox=True, framealpha=0.5,
           prop={'weight':'bold'})

    ax = plt.gca()
    ax.spines['top'].set_linewidth(3)  # 调整上边框线宽
    ax.spines['right'].set_linewidth(3)  # 调整右边框线宽
    ax.spines['left'].set_linewidth(3)  # 调整左边框线宽
    ax.spines['bottom'].set_linewidth(3)

    plt.xticks(fontsize=26, fontweight='bold')
    plt.yticks(fontsize=25, fontweight='bold')

    plt.subplot(4, 1, 4)
    plt.plot(time_steps, current_sequence, color='y', linewidth=3)
    plt.xlabel('Time Step(s)', fontweight="bold", fontsize=25)
    plt.title("Current(A)", fontweight="bold", fontsize=28)
    # plt.ylabel("Current(A)", fontweight="bold", fontsize=18)
    plt.legend(fontsize=35, loc='upper right', fancybox=True, framealpha=0.5,
           prop={'weight':'bold'})

    ax = plt.gca()
    ax.spines['top'].set_linewidth(3)  # 调整上边框线宽
    ax.spines['right'].set_linewidth(3)  # 调整右边框线宽
    ax.spines['left'].set_linewidth(3)  # 调整左边框线宽
    ax.spines['bottom'].set_linewidth(3)

    plt.xticks(fontsize=26, fontweight='bold')
    plt.yticks(fontsize=23, fontweight='bold')

    plt.subplots_adjust(hspace=1, left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig("Chen2020 305K", dpi=300)
    plt.show()


def plot_total_rewards(scores_list):
    """
    绘制每轮的总奖励曲线，并标记每个数据点
    :param scores_list: 每轮的总奖励列表
    """
    episodes = range(1, len(scores_list) + 1)
    plt.plot(episodes, scores_list, linestyle='-', color='blue', label='Total Rewards')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards Over Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()

def copy_agent(agent):
    new_agent = Agent(state_size=3, action_size=1, random_seed=0)
    # Copy actor network parameters
    new_agent.actor_local.load_state_dict(agent.actor_local.state_dict())
    new_agent.actor_target.load_state_dict(agent.actor_target.state_dict())
    new_agent.actor_optimizer.load_state_dict(agent.actor_optimizer.state_dict())

    # Copy critic network parameters
    new_agent.critic_local.load_state_dict(agent.critic_local.state_dict())
    new_agent.critic_target.load_state_dict(agent.critic_target.state_dict())
    new_agent.critic_optimizer.load_state_dict(agent.critic_optimizer.state_dict())

    new_agent.noise = copy.deepcopy(agent.noise)
    new_agent.memory = copy.deepcopy(agent.memory)

    return new_agent

def reptile(agent, param, m, meta_lr):
    n_random_generator = random.Random()
    n_random_generator.seed(None)
    n = (n_random_generator.randint(0, 3))

    init_critic_params = {name: param.clone() for name, param in agent.critic_local.named_parameters()}
    init_actor_params = {name: param.clone() for name, param in agent.actor_local.named_parameters()}
    score_lists = inner_ddpg(agent, n_episodes = m, param = param[n])
    for name, param in agent.critic_local.named_parameters():
        param.data = init_critic_params[name] + meta_lr * (param.data - init_critic_params[name])
    for name, param in agent.actor_local.named_parameters():
        param.data = init_actor_params[name] + meta_lr * (param.data - init_actor_params[name])

    agent.soft_update(agent.critic_local, agent.critic_target, TAU)
    agent.soft_update(agent.actor_local, agent.actor_target, TAU)

    return score_lists

def save_agent_parameters(agent, filename):
    torch.save(
        {
            'actor_local_state_dict': agent.actor_local.state_dict(),
            'actor_target_state_dict': agent.actor_target.state_dict(),
            'critic_local_state_dict': agent.critic_local.state_dict(),
            'critic_target_state_dict': agent.critic_target.state_dict(),
        }, filename
    )

def load_agent_parameters(agent, filename):
    checkpoint = torch.load(filename)
    agent.actor_local.load_state_dict(checkpoint['actor_local_state_dict'])
    agent.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
    agent.critic_local.load_state_dict(checkpoint['critic_local_state_dict'])
    agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])

initial_conditions = {}
meta_sum_scores_list = []
sum_scores_list = []
all_current_sequence = []




