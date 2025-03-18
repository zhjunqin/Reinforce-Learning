import numpy as np

# 定义网格世界的大小和参数
grid_size = (5, 5)
gamma = 0.9  # 折扣因子
reward_default = -1  # 默认奖励（黄色格子或尝试到边界外）
reward_goal = 1  # 蓝色目标格子的奖励
epsilon = 1e-5  # 收敛阈值

# 初始化奖励矩阵
rewards = np.array(
    [
        [0, 0, 0, 0, 0],  # 第一行
        [0, -1, -1, 0, 0],  # 第二行
        [0, 0, -1, 0, 0],  # 第三行
        [0, -1, 1, -1, 0],  # 第四行
        [0, -1, 0, 0, 0],  # 第五行
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

# 根据图片定义策略（每个格子对应的动作）
policy = {
    (0, 0): "right",
    (0, 1): "right",
    (0, 2): "right",
    (0, 3): "down",
    (0, 4): "down",
    (1, 0): "up",
    (1, 1): "up",
    (1, 2): "right",
    (1, 3): "down",
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
value_function = np.zeros(grid_size)


# 辅助函数：获取下一个状态和奖励
def get_next_state_and_reward(state, action):
    i, j = state
    di, dj = actions[action]
    next_i, next_j = i + di, j + dj

    # 检查边界条件，如果越界则保持原地并给予默认奖励
    if next_i < 0 or next_i >= grid_size[0] or next_j < 0 or next_j >= grid_size[1]:
        return state, reward_default

    return (next_i, next_j), rewards[next_i][next_j]


# 贝尔曼方程的迭代求解
def value_iteration():
    global value_function
    while True:
        new_value_function = np.copy(value_function)
        delta = 0
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                state = (i, j)
                action = policy[state]  # 根据策略选择动作
                next_state, reward = get_next_state_and_reward(state, action)
                # 更新价值函数
                new_value_function[i][j] = reward + gamma * value_function[next_state]
                # 更新最大变化量
                delta = max(delta, abs(new_value_function[i][j] - value_function[i][j]))

        value_function = new_value_function
        # 如果变化量小于阈值，则认为收敛，退出迭代
        if delta < epsilon:
            break


# 执行价值迭代
value_iteration()

# 输出最终的价值函数矩阵
print("最终的价值函数矩阵：")
print(np.round(value_function, decimals=1))
