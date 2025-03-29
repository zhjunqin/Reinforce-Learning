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
epsilon = 0.5  # 探索率
num_episodes = 1000  # 采样回合数
max_steps = 10000  # 最大采样步数


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
# 随机初始化策略
policy = {
    (0, 0): np.random.choice(list(action_space.keys())),
    (0, 1): np.random.choice(list(action_space.keys())),
    (0, 2): np.random.choice(list(action_space.keys())),
    (0, 3): np.random.choice(list(action_space.keys())),
    (0, 4): np.random.choice(list(action_space.keys())),
    (1, 0): np.random.choice(list(action_space.keys())),
    (1, 1): np.random.choice(list(action_space.keys())),
    (1, 2): np.random.choice(list(action_space.keys())),
    (1, 3): np.random.choice(list(action_space.keys())),
    (1, 4): np.random.choice(list(action_space.keys())),
    (2, 0): np.random.choice(list(action_space.keys())),
    (2, 1): np.random.choice(list(action_space.keys())),
    (2, 2): np.random.choice(list(action_space.keys())),
    (2, 3): np.random.choice(list(action_space.keys())),
    (2, 4): np.random.choice(list(action_space.keys())),
    (3, 0): np.random.choice(list(action_space.keys())),
    (3, 1): np.random.choice(list(action_space.keys())),
    (3, 2): np.random.choice(list(action_space.keys())),
    (3, 3): np.random.choice(list(action_space.keys())),
    (3, 4): np.random.choice(list(action_space.keys())),
    (4, 0): np.random.choice(list(action_space.keys())),
    (4, 1): np.random.choice(list(action_space.keys())),
    (4, 2): np.random.choice(list(action_space.keys())),
    (4, 3): np.random.choice(list(action_space.keys())),
    (4, 4): np.random.choice(list(action_space.keys())),
}

optimal_policy = {
    (0, 0): "right",
    (0, 1): "right",
    (0, 2): "right",
    (0, 3): "right",
    (0, 4): "down",
    (1, 0): "up",
    (1, 1): "up",
    (1, 2): "right",
    (1, 3): "right",
    (1, 4): "down",
    (2, 0): "up",
    (2, 1): "left",
    (2, 2): "down",
    (2, 3): "right",
    (2, 4): "down",
    (3, 0): "up",
    (3, 1): "right",
    (3, 2): "stay",
    (3, 3): "left",
    (3, 4): "down",
    (4, 0): "up",
    (4, 1): "right",
    (4, 2): "up",
    (4, 3): "left",
    (4, 4): "left",
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
    # 如果到达目标状态，则返回 stay
    if state == dest_state:
        return "stay"
    if np.random.rand() < epsilon:
        return np.random.choice(list(action_space.keys()))
    else:
        return policy[state]


# 按照策略采样一个 episode
# 同时返回统计每个状态-动作对的访问次数
def sample_episode(
    policy, rewards_array, start_state, start_action, dest_state, epsilon, max_steps=1000
):
    episode = []

    # 初始状态访问次数
    state_visit_count = np.zeros((grid_size[0], grid_size[1]))
    # 初始状态动作访问次数
    action_visit_count = np.zeros((grid_size[0], grid_size[1], len(action_space)))

    # 初始化第一个状态-动作对
    next_state, reward = get_next_state_and_reward(start_state, start_action, rewards_array)
    episode.append((start_state, start_action, reward))
    state_visit_count[start_state[0]][start_state[1]] += 1
    action_visit_count[start_state[0]][start_state[1]][action_to_index_mapping[start_action]] += 1
    sample_count = 1

    # 采样一个 episode, 直到到达目标格子或者达到最大步数
    state = next_state
    while sample_count < max_steps:            
        action = sample_action(policy, state, dest_state, epsilon)  # 动作字符串
        next_state, reward = get_next_state_and_reward(state, action, rewards_array)
        
        # 记录所有状态-动作对
        episode.append((state, action, reward))
        # 更新访问次数
        state_visit_count[state[0]][state[1]] += 1
        action_visit_count[state[0]][state[1]][action_to_index_mapping[action]] += 1

        state = next_state
        sample_count += 1
        
    return episode, state_visit_count, action_visit_count


# 给定 episode 反向计算最新的 q_table，使用 gamma 折扣因子
def compute_q_table(episode, gamma):
    G = 0
    return_q_table = np.zeros((grid_size[0], grid_size[1], len(action_space)))
    action_visit_count = np.zeros((grid_size[0], grid_size[1], len(action_space)))

    # 遍历 episode 的反向，从最后一个状态-动作对开始计算回报
    total_samples = len(episode)
    for t in range(total_samples - 1, -1, -1):
        state, action, reward = episode[t]
        G = reward + gamma * G
        
        # 累加所有状态-动作对的回报
        return_q_table[state[0]][state[1]][action_to_index_mapping[action]] += G
        action_visit_count[state[0]][state[1]][action_to_index_mapping[action]] += 1

    # 计算 q_table 的期望
    # 初始值为负无穷
    q_table = np.full((grid_size[0], grid_size[1], len(action_space)), -np.inf)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            for k in range(len(action_space)):
                if action_visit_count[i][j][k] > 0:
                    q_table[i][j][k] = return_q_table[i][j][k] / action_visit_count[i][j][k]
    return q_table


# 给定 q_table 更新 greedy 策略
def update_policy(last_q_table, q_table):
    # 由于采样可能无法覆盖所有的状态-动作对，所以需要合并 last_q_table 和 q_table
    policy = {}
    # 遍历每个状态
    new_q_table = last_q_table
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            state = (i, j)
            # 合并 last_q_table 和 q_table， 取其中的最大值
            q_table_combined = np.maximum(last_q_table[state[0]][state[1]], q_table[state[0]][state[1]])
            # 选择 q_table_combined 中最大的动作
            action = np.argmax(q_table_combined)
            policy[state] = index_to_action_mapping[action]
            new_q_table[state[0]][state[1]] = q_table_combined
    return policy, new_q_table


# 计算状态价值
def compute_state_value(policy, q_table, epsilon):
    action_prob = np.zeros((grid_size[0], grid_size[1], len(action_space)))
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            state = (i, j)
            for action in action_space:
                prob = get_action_prob(policy, state, action, epsilon)
                action_prob[state[0]][state[1]][action_to_index_mapping[action]] = prob

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
    # 输出策略矩阵
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

    last_q_table = np.full((grid_size[0], grid_size[1], len(action_space)), -np.inf)
    for episode_num in range(num_episodes):
        # 随机选择一个起始位置
        start_state = (
            np.random.randint(0, grid_size[0]),
            np.random.randint(0, grid_size[1]),
        )
        # 随机选择一个起始 action
        start_action = np.random.choice(list(action_space.keys()))

        # 更新 epsilon，指数衰减
        epsilon = max(0.01, epsilon * 0.996)
        episode, state_visit_count, action_visit_count = sample_episode(
            policy, rewards_array, start_state, start_action, (3, 2), epsilon, max_steps
        )

        q_table = compute_q_table(episode, gamma)
        policy, last_q_table = update_policy(last_q_table, q_table)

        # 每间隔 100 回合，打印一次策略矩阵和对应的状态价值
        if (episode_num + 1) % 100 == 0:
            print(f"回合 {episode_num + 1} epsilon={round(epsilon, 3)} 的策略矩阵：")
            print_policy(policy)

    return q_table, policy, epsilon

# 获取给定 action 的概率
def get_action_prob(policy, state, action, epsilon):
    if action == policy[state]:
        return epsilon / len(action_space) + (1 - epsilon)
    else:
        return epsilon / len(action_space)

# 根据贝尔曼方程迭代求解策略的状态价值
def compute_state_value_by_bellman(policy, epsilon=0.1, min_delta=0.001):
    value_function = np.zeros(grid_size)
    while True:
        # 这里的 new_value_function 是贝尔曼方程的左边
        new_value_function = np.zeros(grid_size)
        delta = 0
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                state = (i, j)
                for action in action_space:
                    # 获取 action 的概率
                    action_prob = get_action_prob(policy, state, action, epsilon)
                    # 获取下一个状态和奖励
                    next_state, reward = get_next_state_and_reward(state, action, rewards_array)
                    # 更新价值函数
                    new_value_function[i][j] += action_prob * (reward + gamma * value_function[next_state])

                # 更新最大变化量
                delta = max(delta, abs(new_value_function[i][j] - value_function[i][j]))

        value_function = new_value_function
        # 如果变化量小于阈值，则认为收敛，退出迭代
        if delta < min_delta:
            break

    print(f"=== epsilon: {epsilon} ===")
    print(np.round(value_function, decimals=2))
    return value_function

print("最优策略矩阵：")
print_policy(optimal_policy)
print("最优策略在不同 epsilon 下的状态价值矩阵：")
for test_epsilon in [0, 0.1, 0.2, 0.5]:
    compute_state_value_by_bellman(optimal_policy, test_epsilon)

# 测试
q_table, finial_policy, finial_epsilon = monte_carlo(
    policy, epsilon, num_episodes, max_steps
)

print("初始策略矩阵：")
print_policy(policy)
print(f"初始策略矩阵的状态价值矩阵 epsilon= 0：")
compute_state_value_by_bellman(policy, epsilon=0)
print("最终的策略矩阵：")
print_policy(finial_policy)
print(f"最终策略矩阵的状态价值矩阵 epsilon= 0：")
compute_state_value_by_bellman(finial_policy, epsilon=0)
