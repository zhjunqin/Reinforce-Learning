# 蒙特卡罗方法

import numpy as np
import copy
import matplotlib.pyplot as plt

# 定义网格世界的大小和参数
grid_size = (5, 5)
gamma = 0.9  # 折扣因子
reward_boundary = -1.0  # 尝试到边界外奖励
rf = reward_forbidden = -10.0  # 禁止格子的奖励
rg = reward_goal = 1.0  # 蓝色目标格子的奖励
epsilon = 1  # 探索率
num_episodes = 1000  # 采样回合数
max_steps = 100000  # 最大采样步数


# 初始化奖励矩阵
rewards_array = np.array(
    [
        [0, 0, 0, 0, 0],  # 第一行
        [0, rf, rf, 0, 0],  # 第二行
        [0, 0, rf, 0, 0],  # 第三行
        [0, rf, rg, rf, 0],  # 第四行
        [0, rf, 0, 0, 0],  # 第五行
    ]
)

# 定义动作空间
action_space = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "stay": (0, 0),
}

action_to_arrow_mapping = {
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
    "stay": "o",
}

index_to_action_mapping = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
    4: "stay",
}

action_to_index_mapping = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "stay": 4,
}

# 定义初始策略（每个格子对应的动作）
policy = {
    (0, 0): "stay",
    (0, 1): "stay",
    (0, 2): "stay",
    (0, 3): "stay",
    (0, 4): "stay",
    (1, 0): "stay",
    (1, 1): "stay",
    (1, 2): "stay",
    (1, 3): "stay",
    (1, 4): "stay",
    (2, 0): "stay",
    (2, 1): "stay",
    (2, 2): "stay",
    (2, 3): "stay",
    (2, 4): "stay",
    (3, 0): "stay",
    (3, 1): "stay",
    (3, 2): "stay",
    (3, 3): "stay",
    (3, 4): "stay",
    (4, 0): "stay",
    (4, 1): "stay",
    (4, 2): "stay",
    (4, 3): "stay",
    (4, 4): "stay",
}

# 初始化价值函数矩阵
value = np.zeros(grid_size)

# 初始化动作价值函数矩阵
q_table = np.zeros((grid_size[0], grid_size[1], len(action_space)))


# 辅助函数：获取下一个状态和奖励
def get_next_state_and_reward(
    state, action, rewards_array
) -> tuple[tuple[int, int], int]:
    i, j = state
    di, dj = action_space[action]
    next_i, next_j = i + di, j + dj

    # 检查边界条件，如果越界则保持原地并给予默认奖励
    if next_i < 0 or next_i >= grid_size[0] or next_j < 0 or next_j >= grid_size[1]:
        return state, reward_boundary

    return (next_i, next_j), rewards_array[next_i][next_j]


# epsilon-greedy 动作抽样函数，给定策略，随机选择动作
# 返回动作字符串，up, down, left, right, stay
def sample_action(policy, state, dest_state, epsilon) -> str:
    # if state == dest_state:
    #     return "stay"

    if np.random.rand() < epsilon:
        return np.random.choice(list(action_space.keys()))
    else:
        return policy[state]


# 按照策略采样一个 episode
# 同时返回统计每个状态-动作对的访问次数
def sample_episode(
    policy, rewards_array, start_state, dest_state, epsilon, max_steps=1000
):
    episode = []
    state = start_state
    # 初始状态访问次数
    state_visit_count = np.zeros((grid_size[0], grid_size[1]))
    # 初始状态动作访问次数
    action_visit_count = np.zeros((grid_size[0], grid_size[1], len(action_space)))

    # 采样一个 episode, 直到到达目标格子或者达到最大步数
    # while state != dest_state and len(episode) < max_steps:
    sample_count = 0
    while sample_count < max_steps:
        action = sample_action(policy, state, dest_state, epsilon)  # 动作字符串
        next_state, reward = get_next_state_and_reward(state, action, rewards_array)
        episode.append((state, action, reward))
        # 更新访问次数
        state_visit_count[state[0]][state[1]] += 1
        action_visit_count[state[0]][state[1]][action_to_index_mapping[action]] += 1
        state = next_state
        sample_count += 1
    return episode, state_visit_count, action_visit_count


# 给定 episode 反向计算最新的 q_table，使用 gamma 折扣因子
def compute_q_table(episode, gamma, return_q_table, action_visit_count):
    # 初始化 q_table 的累加回报均值
    q_table = np.zeros((grid_size[0], grid_size[1], len(action_space)))
    G = 0

    # 遍历 episode 的反向，从最后一个状态-动作对开始计算回报
    total_samples = len(episode)
    for t in range(total_samples - 1, -1, -1):
        state, action, reward = episode[t]
        G = reward + gamma * G
        return_q_table[state[0]][state[1]][action_to_index_mapping[action]] += G
        action_visit_count[state[0]][state[1]][action_to_index_mapping[action]] += 1

    # 计算 q_table 的期望
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            for k in range(len(action_space)):
                if action_visit_count[i][j][k] > 0:
                    q_table[i][j][k] = (
                        return_q_table[i][j][k] / action_visit_count[i][j][k]
                    )
    return q_table


# 给定 q_table 更新 greedy 策略
def update_policy(q_table):
    policy = {}
    # 遍历每个状态
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            state = (i, j)
            # 选择 q_table 中最大的动作
            action = np.argmax(q_table[state[0]][state[1]])
            policy[state] = index_to_action_mapping[action]
    return policy


# 计算状态价值
def compute_state_value(policy, q_table, epsilon):
    action_prob = np.zeros((grid_size[0], grid_size[1], len(action_space)))
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            state = (i, j)
            for action in action_space:
                if action == policy[state]:
                    # 如果 action 是当前策略，则概率为 epsilon / len(action_space) + (1 - epsilon)
                    action_prob[state[0]][state[1]][action_to_index_mapping[action]] = (
                        epsilon / len(action_space) + (1 - epsilon)
                    )
                else:
                    # 如果 action 不是当前策略，则概率为 epsilon / len(action_space)
                    action_prob[state[0]][state[1]][action_to_index_mapping[action]] = (
                        epsilon / len(action_space)
                    )

    value = np.zeros(grid_size)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            state = (i, j)
            for action in action_space:
                value[i][j] += (
                    action_prob[state[0]][state[1]][action_to_index_mapping[action]]
                    * q_table[state[0]][state[1]][action_to_index_mapping[action]]
                )
    return value


def print_policy(policy):
    # 输出最终的策略矩阵
    print("最终的策略矩阵：")
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            print(f"{action_to_arrow_mapping[policy[(i, j)]]:<5}", end="")
        print()


def print_state_visit_count(state_visit_count):
    # 打印 state 访问次数，使用 matplotlib 绘制点状图
    plt.figure(figsize=(10, 10))
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # 将 (i, j) 转换为 25 个动作的索引
            index = i * 5 + j
            plt.scatter(index, state_visit_count[i][j], c="red", marker="o")
    plt.colorbar()
    plt.title("state visit count")
    plt.show()


# 打印 state-action 访问次数，使用 matplotlib 绘制点状图
def print_action_visit_count(action_visit_count):
    # action_visit_count 的形状是 (5, 5, 5)
    # 显示每个动作的访问次数，横轴为总共 125 个动作，纵轴为访问的次数

    plt.figure(figsize=(10, 10))
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            for k in range(len(action_space)):
                # 将 (i, j, k) 转换为 125 个动作的索引
                index = i * 5 * 5 + j * 5 + k
                plt.scatter(index, action_visit_count[i][j][k], c="red", marker="o")
    plt.colorbar()
    plt.title("state-action visit count")
    plt.show()


# 蒙特卡罗方法
def monte_carlo(policy, epsilon, num_episodes, max_steps):
    # 初始状态访问次数
    state_visit_count = np.zeros((grid_size[0], grid_size[1]))
    # 初始化累加回报 q_table
    return_q_table = np.zeros((grid_size[0], grid_size[1], len(action_space)))
    # 初始化状态-动作访问次数
    action_visit_count = np.zeros((grid_size[0], grid_size[1], len(action_space)))

    for episode_num in range(num_episodes):
        # 随机选择一个起始位置
        start_state = (
            np.random.randint(0, grid_size[0]),
            np.random.randint(0, grid_size[1]),
        )
        # print(f"第 {episode_num} 次采样，起始位置：{start_state}")
        # 当 episode_num 增大，则不断减少 epsilon，增加探索
        epsilon = max(epsilon * 0.999, 0.1)
        episode, current_state_visit_count, current_action_visit_count = sample_episode(
            policy, rewards_array, start_state, (3, 2), epsilon, max_steps
        )
        # 更新状态访问次数
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                state_visit_count[i][j] += current_state_visit_count[i][j]
        # 更新状态动作访问次数
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                for k in range(len(action_space)):
                    action_visit_count[i][j][k] += current_action_visit_count[i][j][k]

        q_table = compute_q_table(episode, gamma, return_q_table, action_visit_count)
        policy = update_policy(q_table)
        # 每间隔一定的 episode 打印 policy
        # if episode_num % 1000 == 0:
        #     print(f"\n第 {episode_num} 次采样后的策略：")
        #     print_policy(policy)
    return q_table, policy, state_visit_count, action_visit_count


# 测试
q_table, policy, state_visit_count, action_visit_count = monte_carlo(
    policy, epsilon, num_episodes, max_steps
)
state_value = compute_state_value(policy, q_table, epsilon)


print_policy(policy)
print("=== state visit count ===")
print(state_visit_count)
print("=== action visit count ===")
print(action_visit_count)
print("=== state value ===")
print(state_value)

# print_state_visit_count(state_visit_count)
# print_action_visit_count(action_visit_count)
