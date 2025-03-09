# 状态价值（State Values）和贝尔曼方程（Bellman Equation）

## 状态价值（State Values）

为了介绍状态价值先引入一些符号。

考虑一个时间步骤序列 $t = 0, 1, 2, ...$，在时刻 $t$，Agent 处于状态 $ S_t$，并按照策略 $\pi$ 采取了动作 $A_t$。然后转移到下一个状态 $S_{t+1}$，获得的即时奖励是 $R_{t+1}$。这个过程可以简洁的表示为

$$S_t \xrightarrow{A_t}  S_{t+1},R_{t+1}$$

注意，这里的 $S_t,S_{t+1},A_t,R_{t+1}$ 都是随机变量，而且  $S_t,S_{t+1} \in \mathcal{S}$， $A_t \in \mathcal{A(S_t)}$，$R_{t+1} \in \mathcal{R(S_t, A_t)}$

从 $t$ 时刻开始，可以获得一个 state-action-reward 的轨迹：

$$ S_t \xrightarrow{A_t} S_{t+1},R_{t+1} \xrightarrow{A_{t+1}} S_{t+2},R_{t+2} \xrightarrow{A_{t+2}} S_{t+3},R_{t+3} ... $$

沿着轨迹，获得的折扣回报是从时刻 $t$ 开始的累积折扣奖励：

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...$$

$\gamma$ 是折扣因子，$\gamma \in (0, 1) $。

注意，这里的 $R_{t+1}, R_{t+2}, ...$ 都是随机变量，因此 $G_t$ 也是随机变量。

由于 $G_t$ 是随机变量，我们可以计算它的期望值：

$$v_\pi(s) = \mathbb{E}_\pi[G_t|S_t=s]$$

理解为从状态 $s$ 开始，按照策略 $\pi$ 行动，获得的期望回报。

这里 $v_\pi(s)$ 称之为**状态价值函数(state-value function)** 或者 $s$ 的**状态价值(state value)**。

一些重要的说明：
- $v_\pi(s)$ 依赖于 $s$。这是因为它的定义是一个条件期望，条件是 Agent 从 $S_t=s$ 开始。
- $v_\pi(s)$ 依赖于 $\pi$。这是因为轨迹是通过遵循策略 $\pi$ 生成的。对于不同的策略，状态价值可能不同。
- $v_\pi(s)$ 不依赖于 $t$。如果 Agent 在状态空间中移动，$t$ 代表当前时间步。一旦给出策略 $\pi$，其对应的$v_\pi(s)$ 值就确定了，与时刻 $t$ 无关。

### 状态价值与回报之间的关系

当策略和系统模型都是确定的时（采取的策略固定，迁移的状态固定），从某个状态 $s$ 开始总是会有相同的轨迹。在这种情况下，从一个状态 $s$ 开始获得的回报等于该状态的价值。

相比之下，当策略或系统模型中的任何一个都是随机的时，从相同的状态 $s$ 开始可能会生成不同的轨迹。在这种情况下，不同轨迹的回报是不同的，状态价值是这些回报的平均值。

尽管回报可以用来评估策略，但使用状态价值来评估策略更为正式：产生更高状态价值的策略更好。因此，状态价值构成了强化学习中的一个核心概念。

## 贝尔曼方程（Bellman Equation）

接下来推导贝尔曼方程。
首先，我们可以将折扣回报 $G_t$ 重写为：

$$\begin{aligned}
G_t &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... \\
&= R_{t+1} + \gamma(R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + ...) \\
&= R_{t+1} + \gamma G_{t+1} \\
\end{aligned}$$

这个等式建立了 $G_t$ 和 $G_{t+1}$之间的关系。表明折扣回报 $G_t$ 可以分解为即时奖励 $R_{t+1}$ 和下一个时刻的折扣回报 $G_{t+1}$ 的加权和。

因此状态价值函数可以被重写为：

$$\begin{aligned}
v_\pi(s) &= \mathbb{E}_\pi[G_t|S_t=s] \\
&= \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1}|S_t=s] \\
&= \mathbb{E}_\pi[R_{t+1} |S_t=s] +  \gamma \mathbb{E}_\pi[G_{t+1}|S_t=s]
\end{aligned}$$

- 方程的第一项 $\mathbb{E}_\pi[R_{t+1} |S_t=s]$ 是**即时奖励的期望值**。使用全期望定理，可以计算为：
  $$\begin{aligned}
  \mathbb{E}_\pi[R_{t+1} |S_t=s] &= \sum_{a \in \mathcal{A}} \pi(a|s) \mathbb{E}_\pi[R_{t+1} |S_t=s, A_t=a] \\
  &= \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{r \in \mathcal{R}} p(r|s,a)r \\
  \end{aligned}$$
  这里 $\mathcal{A}$ 和 $\mathcal{R}$ 分别是所有可能的动作和奖励的集合。需要注意的是，对于不同的状态，$\mathcal{A}$ 可能不同。在这种情况下，$\mathcal{A}$ 应写作 $\mathcal{A(s)}$。同样，$\mathcal{R}$ 也可能依赖于 $(s, a)$。这里省略了 $s$ 或 $(s, a)$ 的依赖。
- 方程的第二项 $\mathbb{E}_\pi[G_{t+1}|S_t=s]$ 是**未来奖励的期望值**。使用全期望定理，可以计算为：
  $$\begin{aligned}
  \mathbb{E}_\pi[G_{t+1}|S_t=s] &= \sum_{a \in \mathcal{A}} \pi(a|s) \mathbb{E}_\pi[G_{t+1}|S_t=s, A_t=a] \\
  &= \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}} p(s'|s,a) \mathbb{E}_\pi[G_{t+1}|S_{t+1}=s'] \\
  &= \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}} p(s'|s,a) v_\pi(s')
  \end{aligned}$$
  这里 $\mathcal{S}$ 是所有可能的状态的集合。第二步的转换利用到了马尔可夫性质 $\mathbb{E}_\pi[G_{t+1}|S_t=s, A_t=a, S_{t+1}=s'] = \mathbb{E}_\pi[G_{t+1}|S_{t+1}=s']$。最后一个等式是因为 $\mathbb{E}_\pi[G_{t+1}|S_{t+1}=s']$ 就是状态 $s'$ 的状态价值 $v_\pi(s')$。
将上面两个等式代入到

  $$\begin{aligned}
  v_\pi(s) &= \mathbb{E}_\pi[R_{t+1} |S_t=s] +  \gamma \mathbb{E}_\pi[G_{t+1}|S_t=s] \\
  &= \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{r \in \mathcal{R}} p(r|s,a)r +  \gamma \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}} p(s'|s,a) v_\pi(s') \\
  &= \sum_{a \in \mathcal{A}} \pi(a|s) \left(\sum_{r \in \mathcal{R}} p(r|s,a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) v_\pi(s')\right), \forall s \in \mathcal{S}
  \end{aligned}$$

这个方程就是**贝尔曼方程（Bellman Equation）**。它描述了状态价值之间的关系。它是设计和分析强化学习算法的基本工具。

方程说明如下：
- $v_\pi(s)$ 和 $v_\pi(s')$ 都是待计算的状态价值。这里可能会感到疑惑，因为一个未知的值 $v_\pi(s)$ 依赖于另外一个未知量 $v_\pi(s')$。这里需要注意的是贝尔曼方程是所有状态的一组线性方程，而不是单个方程。当把所有的方程组合起来，就会清楚如何计算所有的状态价值。
- $\pi(s)$是一个给定的策略。由于状态价值可以用来评估策略，从贝尔曼方程中求解状态价值是一个策略评估的过程，这是许多强化学习算法中的重要过程。
- $p(r|s,a)$ 和 $p(s'|s,a)$ 表示系统模型。这里的计算需要依赖于知道这些系统模型的概率，也就是知道了系统模型（环境）状态迁移的概率和奖励的概率，因此称之为 **model-based** 的强化学习。后面还会介绍 **model-free** 的强化学习方法。**model-free** 方法不需要事先知道系统的这些概率。



## 参考文献
- https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning
- 