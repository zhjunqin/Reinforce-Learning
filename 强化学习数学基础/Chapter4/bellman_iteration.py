import numpy as np
import copy

# 定义网格世界的大小和参数
grid_size = (5, 5)
gamma = 0.9  # 折扣因子
reward_boundary = -1  # 尝试到边界外奖励
rf = reward_forbidden = -10  # 禁止格子的奖励
rg = reward_goal = 1  # 蓝色目标格子的奖励
epsilon = 1e-5  # 收敛阈值

# 初始化奖励矩阵
rewards = np.array(
    [
        [0, 0, 0, 0, 0],  # 第一行
        [0, rf, rf, 0, 0],  # 第二行
        [0, 0, rf, 0, 0],  # 第三行
        [0, rf, rg, rf, 0],  # 第四行
        [0, rf, 0, 0, 0],  # 第五行
    ]
)

# 定义动作空间
actions = {
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

# 根据图片定义策略（每个格子对应的动作）
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
q_table = np.zeros((grid_size[0], grid_size[1], len(actions)))


# 辅助函数：获取下一个状态和奖励
def get_next_state_and_reward(state, action):
    i, j = state
    di, dj = actions[action]
    next_i, next_j = i + di, j + dj

    # 检查边界条件，如果越界则保持原地并给予默认奖励
    if next_i < 0 or next_i >= grid_size[0] or next_j < 0 or next_j >= grid_size[1]:
        return state, reward_boundary

    return (next_i, next_j), rewards[next_i][next_j]


# 贝尔曼价值迭代
def bellman_value_iteration(value_k, policy, epsilon, j_truncated=-1):
    j_iter = 0
    while True:
        value_k_1 = np.copy(value_k)
        delta = 0
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                state = (i, j)
                action = policy[state]  # 根据策略选择动作
                next_state, reward = get_next_state_and_reward(state, action)
                # 更新价值函数
                value_k_1[i][j] = reward + gamma * value_k[next_state]
                # 更新最大变化量
                delta = max(delta, abs(value_k_1[i][j] - value_k[i][j]))

        value_k = value_k_1
        # 如果变化量小于阈值，则认为收敛，退出迭代
        if delta < epsilon:
            break
        j_iter += 1
        if j_truncated > 0 and j_iter >= j_truncated:
            break
    return value_k


# 价值迭代算法
def value_iteration(value_0, policy_0, q_table):
    value_k = np.copy(value_0)
    policy_k = copy.deepcopy(policy_0)
    q_table_k = np.copy(q_table)
    iteration_count = 0
    while True:
        value_k_1 = np.copy(value_k)
        delta = 0
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                state = (i, j)
                for action in actions:
                    next_state, reward = get_next_state_and_reward(state, action)
                    # 更新动作价值函数
                    q_table_k[i][j][action_to_index_mapping[action]] = (
                        reward + gamma * value_k[next_state]
                    )
                # 获取最大动作价值函数和对应的动作
                max_q_value = np.max(q_table_k[i][j])
                max_q_action = np.argmax(q_table_k[i][j])
                # 更新策略
                policy_k[state] = index_to_action_mapping[max_q_action]
                # 更新状态价值函数
                value_k[i][j] = max_q_value
                # 更新最大变化量
                delta = max(delta, abs(value_k[i][j] - value_k_1[i][j]))

        # 如果变化量小于阈值，则认为收敛，退出迭代
        iteration_count += 1
        if delta < epsilon:
            break

    return value_k, policy_k, q_table_k, iteration_count


# 策略迭代算法
def policy_iteration(value_0, policy_0, q_table):
    value_k = np.copy(value_0)
    policy_k = copy.deepcopy(policy_0)
    q_table_k = np.copy(q_table)
    iteration_count = 0
    while True:
        value_k_1 = np.copy(value_k)
        delta = 0
        # 价值迭代
        value_k = bellman_value_iteration(value_k, policy_k, epsilon)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                state = (i, j)
                for action in actions:
                    next_state, reward = get_next_state_and_reward(state, action)
                    # 更新动作价值函数
                    q_table_k[i][j][action_to_index_mapping[action]] = (
                        reward + gamma * value_k[next_state]
                    )
                # 获取最大动作价值函数和对应的动作
                max_q_value = np.max(q_table_k[i][j])
                max_q_action = np.argmax(q_table_k[i][j])
                # 更新策略
                policy_k[state] = index_to_action_mapping[max_q_action]
                # 更新状态价值函数
                value_k[i][j] = max_q_value
                # 更新最大变化量
                delta = max(delta, abs(value_k[i][j] - value_k_1[i][j]))

        # 如果变化量小于阈值，则认为收敛，退出迭代
        iteration_count += 1
        if delta < epsilon:
            break

    return value_k, policy_k, q_table_k, iteration_count


# 截断策略迭代算法
def truncted_policy_iteration(value_0, policy_0, q_table, j_truncated):
    value_k = np.copy(value_0)
    policy_k = copy.deepcopy(policy_0)
    q_table_k = np.copy(q_table)
    iteration_count = 0
    while True:
        value_k_1 = np.copy(value_k)
        delta = 0
        # 价值迭代
        value_k = bellman_value_iteration(
            value_k, policy_k, epsilon, j_truncated=j_truncated
        )
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                state = (i, j)
                for action in actions:
                    next_state, reward = get_next_state_and_reward(state, action)
                    # 更新动作价值函数
                    q_table_k[i][j][action_to_index_mapping[action]] = (
                        reward + gamma * value_k[next_state]
                    )
                # 获取最大动作价值函数和对应的动作
                max_q_value = np.max(q_table_k[i][j])
                max_q_action = np.argmax(q_table_k[i][j])
                # 更新策略
                policy_k[state] = index_to_action_mapping[max_q_action]
                # 更新状态价值函数
                value_k[i][j] = max_q_value
                # 更新最大变化量
                delta = max(delta, abs(value_k[i][j] - value_k_1[i][j]))

        # 如果变化量小于阈值，则认为收敛，退出迭代
        iteration_count += 1
        if delta < epsilon:
            break

    return value_k, policy_k, q_table_k, iteration_count


def print_result(value_k, policy_k):
    # 输出最终的价值函数矩阵
    print("最终的价值函数矩阵：")
    print(np.round(value_k, decimals=1))
    # 将策略矩阵按照每行每列打印，并且按照每个格子占6个字符对齐
    print("最终的策略矩阵：")
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            print(f"{action_to_arrow_mapping[policy_k[(i, j)]]:<6}", end="")
        print()


# 输出奖励矩阵
print("========= 奖励矩阵 ==========")
print(rewards)

# 执行价值迭代
print("========= 价值迭代 ==========")
value_k, policy_k, q_table_k, iteration_count = value_iteration(value, policy, q_table)
print(f"迭代次数：{iteration_count}")
print_result(value_k, policy_k)

# 执行策略迭代
print("========= 策略迭代 ==========")
value_k, policy_k, q_table_k, iteration_count = policy_iteration(value, policy, q_table)
print(f"迭代次数：{iteration_count}")
print_result(value_k, policy_k)

# 执行截断策略迭代
print("========= 截断策略迭代 ==========")
value_k, policy_k, q_table_k, iteration_count = truncted_policy_iteration(
    value, policy, q_table, j_truncated=8
)
print(f"迭代次数：{iteration_count}")
print_result(value_k, policy_k)
