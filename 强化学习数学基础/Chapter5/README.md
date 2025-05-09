# 蒙特卡洛方法（Monte Carlo Methods）

之前的章节介绍了基于系统模型来找到最优策略的算法。在这个章节中，我们开始介绍 `model-free` 的强化学习算法。

但我们必须填补一个知识空白：我们如何在没有模型的情况下找到最优策略？

理念很简单：如果我们没有模型，我们必须有一些数据。如果我们没有数据，我们必须有一个模型。如果我们两者都没有，那么我们就无法找到最佳策略。强化学习中的“数据”通常是指智能体与环境的交互经验。

## 均值估计（Mean estimation）

考虑一个随机变量 $X$，它可以从表示为 $\mathcal{X}$ 的有限实数集中获取值。假定我们的任务是计算 $X$ 的平均值或期望： $E[X]$。有两种办法来计算 $E[X]$。

### model-based 的方法

这里的模型是指随机变量 $X$ 的概率分布。如果 $X$ 的概率分布是已知的，那么可以基于期望的定义直接计算出均值：

$$ E[X] = \sum_{x \in \mathcal{X}} p(x) \cdot x$$

在本书中，我们交替使用术语期望值、平均值。

### model-free 的方法

如果随机变量 $X$ 的概率分布是未知的，假定我们有一些 $X$ 的样本 ${x_1, x_2, ..., x_n}$，然后均值可以近似为

$$ E[X] \approx \bar{x} = \frac{1}{n} \sum^n_{j=1} x_j$$

当 $n$ 比较小的时候，近似值可能不准。但是，随着 $n$ 的增加，近似值变得越来越准确。当 $ n \rightarrow \infin $ 时，我们有 $ \bar{x} \rightarrow E[X] $。

这个是大数定律（law of large numbers）保证的：大量样本的均值接近期望值。

### 示例 

考虑一个掷硬币的游戏，让随机变量 $X$ 表示落地时表示哪一面。

则随机变量 $X$ 有两种可能： $X=1$ 时显示正面，$X=-1$ 时显示反面。

假定 $X$ 的真实概率分布（即模型）为

$$ p(X=1) = 0.5, \ p(X=-1) = 0.5 $$

如果概率分布式是已知的，则可以直接计算出平均值：

$$ E[X] = 0.5 \cdot 1 + 0.5 \cdot -1 = 0$$

假如概率分布未知，那么我们可以多次抛硬币并记录抽样结果 $ \{x_i\} ^n_{i=1}$。通过计算抽样的平均值，我们可以获得平均值的估计值。如下图所示，随着样本数量的增加，估计的平均值变得越来越准确。

![](./assets/chapter5_sample_mean.png)

值得一提的是，用于均值估计的样本必须是独立同分布的（i.i.d.）。否则，如果采样值相关，则可能无法正确估计预期值。

## MC Basic：最简单的 MC-based 算法

将策略迭代（policy iteration）转换为 `model-free` 的算法。

回顾策略迭代算法，包含两个步骤，1. 策略评估（policy evaluation），即通过逐个元素迭代的方式来从 $ v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k} $ 中求解 $v_{\pi_k}$；2. 策略改进（policy improvement），旨在计算贪婪策略 $ \pi_{k+1}= \argmax_\pi (r_{\pi} + \gamma P_{\pi} v_{\pi_k}) $，逐个元素的形式为：

$$\begin{aligned}
\pi_{k+1}(s) &= \argmax_\pi \sum_{a} \pi(a|s) \left[\sum_{r} p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a) v_{\pi_k}(s')\right] \\
&= \argmax_\pi \sum_{a} \pi(a|s) q_{\pi_k}(s,a)，\forall s \in S
\end{aligned}$$

在策略迭代（policy iteration）的步骤中，第一步策略评估（policy evaluation）中求解 $v_{\pi_k}$ 的目的是为了计算第二步中的 $q_{\pi_k}(s,a)$ 来做基础。在 `model-based` 的方法中，必须要有 $v_{\pi_k}(s')$ 来计算 $q_{\pi_k}(s,a)$。但是在 `model-free` 的方法中，可以不需要 $v_{\pi_k}(s')$，直接通过采样的方法得到 $q_{\pi_k}(s,a)$ 的估计值。

### model-based 的方法

如之前所述，先通过求解贝尔曼方程来计算出 $v_{\pi_k}(s')$，然后计算得到动作价值 $q_{\pi_k}(s,a)$

$$ q_{\pi_k}(s,a) = \sum_{r} p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a) v_{\pi_k}(s') $$

这里需要知道系统模型 $p(r|s,a), p(s'|s,a)$。

### model-free 的方法

回想一下，动作价值的定义是

$$\begin{aligned}
q_\pi(s,a) &= \mathbb{E}_\pi[G_{t}|S_t=s, A_t=a] \\
&= \mathbb{E}_\pi[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} +...|S_t=s, A_t=a] \\
\end{aligned}$$

也就是从动作 $(s,a)$ 开始时获得的期望回报。由于 $q_\pi(s,a)$ 是一个期望值，因此可以用 MC 方法来估计。

从 $(s,a)$ 开始，Agent 按照策略 $\pi$ 和环境进行互动，从而得到一定数量的回合（episode）样本。假定有 $n$ 个回合，其中第 $i\th$ 回合的回报为 $g^{(i)}_{\pi_k}(s,a)$，则 $q_\pi(s,a)$ 可以近似为：

$$ q_\pi(s,a) = \mathbb{E}_\pi[G_{t}|S_t=s, A_t=a] \approx \frac{1}{n} \sum^n_{i=1} g^{(i)}_{\pi_k}(s,a)  $$

如果 $n$ 足够大，那么根据大数定律，近似值将足够准确。

基于 MC 的强化学习的基本思想是使用 `model-free` 的方法来估计动作值，以取代策略迭代算法中 `model-based` 的方法。

### MC Basic 算法

这里介绍第一个 MC-based 强化学习方法。

从初始策略 $\pi_0$ 开始，该算法在第 $k\th$ 次迭代中有两个步骤：

#### Step 1: 策略评估（Policy evaluation）

这个步骤是对所有的 $(s,a)$ 评估 $q_{\pi_k}(s,a)$。具体地，我们对于所有的 $(s,a)$ 收集足够多的回合，然后用回报的平均值 $q_{k}(s,a)$ 来近似估计 $q_{\pi_k}(s,a)$。

#### Step 2: 策略改进（Policy improvement）

这个步骤是对所有的 $s$，求解 $ \pi_{k+1}(s)=\argmax_\pi \sum_{a} \pi(a|s) q_{k}(s,a)$。贪婪最优策略为 

$$
\pi_{k+1}(a|s) = 
\begin{cases}
1, & a = a^*_k(s), \\
0, & a \neq a^*_k(s).
\end{cases}$$

这里的 $a^*_k(s) = \argmax_a q_k(s,a)$。

伪代码算法如下：

![](./assets/chapter5_mc_basic.png)

但是 MC Basic 由于比较低的采样效率，过于简单而不实用。

## MC Exploring Starts

接下来，我们扩展了 MC Basic 算法，以获得另一种基于 MC 的强化学习算法，该算法稍微复杂一些，但采样效率更高。

基于 MC 的强化学习的一个重要方面是如何更有效地使用样本。具体来说，假设我们有一个回合通过遵循策略 $\pi$ 获得的样本：

$$ s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_4} s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} s_5 \xrightarrow{a_1} ... $$

每次在一个回合中出现 state-action 对时，称为对该 state-action 对的访问（visit）。可以采用不同的策略来利用这些 state-action 对。

### initial visit

第一个也是最简单的策略是使用初次访问（initial visit）。也就是说，仅使用一个回合中初始 state-action 对来估计其对应的动作价值。对于上面的示例，初始访问策略仅使用该回合来估计 $(s_1,a_2)$ 的动作价值。

但是这样的方式无法有效的使用样本，因为一个回合也访问了其他的状态-动作对，比如 $(s_2,a_4), (s_1,a_2), (s_5,a_1)$。我们可以把一个回合拆解为多个子回合。

![](./assets/chapter5_visit.png)

状态-动作对访问后生成的轨迹可以被视为一个新的回合。这些新回合可用于估算更多动作价值。这样，可以更有效地利用回合中的样本。

### first-visit

状态-作对可能会在一次回合中被多次访问。例如，$(s_1, a_2)$ 在一个回合中被访问了两次。如果我们只计算首次被访问的情况，则称为 `first-visit` 策略。

### every-visit

如果我们计算 state-action 对的每次访问的情况，这样的策略称为 `every-visit`。也就是一个回合中如果一个 state-action 对重复出现，则将重复出现的情况做平均。

### 算法描述

因此，得到如下 `MC Exploring Starts` 策略：

![](./assets/chapter5_mc_exploring_starts.png)

值得注意的是，算法中是从一个回合的数据后面往前计算，这样可以提高计算的效率。

`MC Exploring Statts` 策略可以避免 Agent 必须等到收集完所有回合后才能更新估计值，就是使用单个回合的回报来近似相应的动作价值。这样，当我们采集到一个回合的样本后，我们可以立即获得一个粗略的估计。然后，可以逐个回合地改进策略。

但是 `exploring starts` 条件需要从每个 state-action 对开始的足够多的回合。只有对每一个 state-action 对都进行了很好的探索，我们才能准确估计它们的动作价值（根据大数定律），从而成功找到最优策略。但是，在实际中很难满足此条件，尤其是那些涉及与环境物理交互的应用程序。


## MC $\epsilon\text{-greedy}$: 不需要 exploring starts 的学习方式

接下来，我们通过移除 `exploring starts` 条件来扩展 MC Exploring Starts 算法。这个条件其实要求每个 state-action 对都可以被足够多次地访问，这也是基于软策略（soft policies）来实现的。

### $\epsilon\text{-greedy}$ 策略

$\epsilon\text{-greedy}$ 策略是一种随机策略，它有更高的几率选择 greedy 动作，并且采取任何其他作的非零概率相同。在这里，greedy 动作是指具有最大动作价值的动作。假设 $\epsilon \in [0,1]$，相应的 $\epsilon\text{-greedy}$ 策略具有以下形式：

$$
\pi(a|s) =
\begin{cases} 
1 - \frac{\epsilon}{|\mathcal{A}(s)|}(|\mathcal{A}(s)| - 1) = 1 -{\epsilon} + \frac{\epsilon}{|\mathcal{A}(s)|}, & \text{for the greedy action,} \\ 
\frac{\epsilon}{|\mathcal{A}(s)|}, & \text{for the other } |\mathcal{A}(s)| - 1 \text{ actions.}
\end{cases}
$$

这里 $|\mathcal{A}(s)|$ 表示和状态 $s$ 相关联的动作数量。

由于 $\epsilon \in [0,1]$，则可知上面式子的值总是高于下面，因此采取 greedy 动作的概率总是高于其他的动作。

当 $\epsilon = 0$，$\epsilon\text{-greedy}$ 只会采取 greedy 动作。当 $\epsilon = 1$，所有动作的概率都相同为 $\frac{1}{|\mathcal{A}(s)|}$。

如何生成这样的一个随机策略？

可以首先在均匀分布 $[0,1]$ 之间生成一个随机数 $x$，如果 $x \geq \epsilon$，则选择 greedy 动作（$1 - \epsilon$ 的概率）。如果 $x \lt \epsilon$，则在所有的动作中随机选择一个（这个里面包含了 greedy 动作），所有动作的概率都为 $\frac{\epsilon}{|\mathcal{A}(s)|}$。可以表示为：

$$
\pi(a|s) =
\begin{cases} 
 1 -{\epsilon}, & \text{for the greedy action,} \\ 
\frac{\epsilon}{|\mathcal{A}(s)|}, & \text{for all } |\mathcal{A}(s)|  \text{ actions.}
\end{cases}
$$

### 算法描述

$MC\ \epsilon\text{-}Greedy $ 算法如下：

![](./assets/chapter5_epislon_greedy.png)

图片中 `Policy improvement` 应该可以放到循环外。

### 探索（Exploration）和利用（Exploitation）

探索（Exploration）和利用（Exploitation）构成了强化学习的基本权衡（tradeoff）。在这里，**探索**意味着策略可能会执行尽可能多的作。这样，所有的状态-动作都可以被很好地访问和评估。**利用**意味着改进的策略应该采取具有最大动作价值的贪婪动作。但是，由于当前时刻获得的动作价值可能由于探索不足而不准确，我们应该在进行开发的同时不断探索，以免错过最优动作。

$\epsilon\text{-greedy}$ 提供了一个可以平衡探索（Exploration）和利用（Exploitation）的方法。一方面有比较高的概率来选择 greedy 动作（利用已有的经验），另外一方面可以有一定的概率来选择其他动作（探索其他未访问过的动作）。

## 代码实现

详细参考代码 [monte_carlo.py](./monte_carlo.py)。

### 给定策略，计算状态价值

最优策略在不同 $\epsilon$ 下的状态价值矩阵，可以看到随着 $\epsilon$ 的增加，状态价值在降低。

```
最优策略矩阵：
→    →    →    →    ↓
↑    ↑    →    →    ↓
↑    ←    ↓    →    ↓
↑    →    o    ←    ↓
↑    →    ↑    ←    ←

最优策略在不同 epsilon 下的状态价值矩阵：
=== epsilon: 0 ===
[[3.48 3.87 4.3  4.77 5.31]
 [3.13 3.48 4.77 5.31 5.9 ]
 [2.82 2.53 9.99 5.9  6.55]
 [2.53 9.99 9.99 9.99 7.28]
 [2.28 8.99 9.99 8.99 8.09]]
=== epsilon: 0.1 ===
[[ 0.42  0.52  0.87  1.25  1.44]
 [ 0.13  0.04  0.53  1.26  1.66]
 [ 0.08 -0.42  3.36  1.35  1.9 ]
 [-0.1   3.38  3.33  3.66  2.18]
 [-0.29  2.8   3.7   3.06  2.68]]
=== epsilon: 0.2 ===
[[-2.19 -2.36 -2.07 -1.69 -1.82]
 [-2.46 -3.02 -3.29 -2.25 -1.97]
 [-2.28 -3.33 -2.51 -2.78 -2.17]
 [-2.53 -2.49 -2.81 -2.03 -2.38]
 [-2.8  -2.78 -2.06 -2.31 -2.14]]
=== epsilon: 0.5 ===
[[ -7.92  -8.9   -8.37  -7.14  -7.69]
 [ -8.64 -10.76 -12.31  -9.47  -8.83]
 [ -8.21 -12.24 -15.18 -12.2  -10.37]
 [ -9.58 -15.21 -16.87 -14.24 -12.06]
 [-10.6  -15.23 -14.93 -14.07 -12.17]]
```

### MC $\epsilon\text{-greedy}$ 算法

算法执行结果：

```
初始策略矩阵：
↓    ↓    ←    ↑    →
←    ↓    ↑    o    ↑
→    ↓    o    ↓    ↑
o    ↓    ↓    ↑    o
↓    ↓    →    →    →
初始策略矩阵的状态价值矩阵 epsilon= 0：
=== epsilon: 0 ===
[[ -9.   -31.95 -28.76 -10.   -10.  ]
 [-10.   -24.39 -25.88   0.    -9.  ]
 [-24.39 -27.1  -99.99 -52.63  -8.1 ]
 [  0.   -19.    -7.29 -47.36   0.  ]
 [-10.   -10.    -8.1   -9.   -10.  ]]
最终的策略矩阵：
→    →    →    ↓    ↓
↑    ↑    →    ↓    ↓
↑    ←    ↓    →    ↓
↑    →    o    ←    ↓
↑    →    ↑    ←    ←
最终策略矩阵的状态价值矩阵 epsilon= 0：
=== epsilon: 0 ===
[[3.48 3.87 4.3  4.77 5.31]
 [3.13 3.48 4.77 5.31 5.9 ]
 [2.82 2.53 9.99 5.9  6.55]
 [2.53 9.99 9.99 9.99 7.28]
 [2.28 8.99 9.99 8.99 8.09]]
```

收敛过程如下，大部分情况下，在前 500 次迭代的时候都会收敛到最优策略。

```
回合 100 epsilon=0.335 的策略矩阵：
→    →    →    →    ↓
↑    ↑    →    ↓    ↓
↓    ←    ↓    →    ↓
↑    →    o    ←    ↓
↑    →    ↑    ←    ←
回合 200 epsilon=0.224 的策略矩阵：
→    →    →    →    ↓
↑    ↑    →    ↓    ↓
↑    ←    ↓    →    ↓
↑    →    o    ←    ↓
↑    →    ↑    ←    ←
回合 300 epsilon=0.15 的策略矩阵：
→    →    →    →    ↓
↑    ↑    →    ↓    ↓
↑    ←    ↓    →    ↓
↑    →    o    ←    ↓
↑    →    ↑    ←    ←
回合 400 epsilon=0.101 的策略矩阵：
→    →    →    ↓    ↓
↑    ↑    →    ↓    ↓
↑    ←    ↓    →    ↓
↑    →    o    ←    ↓
↑    →    ↑    ←    ←
回合 500 epsilon=0.067 的策略矩阵：
→    →    →    ↓    ↓
↑    ↑    →    ↓    ↓
↑    ←    ↓    →    ↓
↑    →    o    ←    ↓
↑    →    ↑    ←    ←
```

### 讨论

值得注意的一些地方。
- 使用了 $\epsilon$ 指数衰减的方式
  - `epsilon = max(0.01, epsilon * 0.996)` 
  - 前期偏探索，后期偏利用
- $\text{Return}(s,a)$ 和 $\text{Num}(s,a)$ 的处理
  - 在每个迭代中重新赋值为 0。
  - 如果将所有的迭代累加，之前的回报会不断累积，导致后面新的回报影响非常小，可能进入了一种局部收敛的情况。
- 样本的采样方式
  - 当到达目标状态后，只返回了 `stay` 的动作。
  - 如果不返回 `stay` 的动作，从实验的情况来看会导致无法收敛。
  - 可能是由于，这样在计算回报的时候，其实回报不是到最终态的累加。
- 在 `Policy evaluation` 步骤中，$q(s_t, a_t)$ 的处理
  - 将初始值全部赋值为 $-\infin$。
  - 由于采样可能无法覆盖到所有的 `state-action` 对，会出现部分 `state-action` 动作价值为 $-\infin$ 的情况，导致这个时候取 $\argmax_a q(s_t, a_t)$ 会不准确。
  - 特别是当  $\epsilon$ 衰减到最小的时候。
  - 因此保留了历史最大的 $q(s_t, a_t)$。

经过这些处理后，收敛的速度有明显的提升。