# 最优状态价值和贝尔曼最优方程

强化学习的最终目标是寻求最优策略。因此，有必要定义什么是最优策略。在本章中，我们介绍了一个核心概念和一个重要工具。核心概念是最优状态价值，基于此我们可以定义最优策略。重要工具是贝尔曼最优方程，通过它可以求解最优状态价值和策略。

## 最优策略和最优状态价值

**定义：** **最优策略和最优状态价值 (Optimal policy and optimal state value)**

一个策略 $\pi^*$ 是最优的，那么它对于所有的策略 $\pi$ 在所有的状态 $s$ 下满足:

$$ v_{\pi^*}(s) \geq v_{\pi}(s), \quad \forall s \in \mathcal{S}  $$

则称之为**最优策略**。同时最优策略 $\pi^*$ 的状态价值是**最优状态价值**。

上述定义表明，最优策略相对于所有其他策略，在每个状态下都具有最大的状态价值。这个定义也引发了许多问题：
- 存在：最优策略是否存在？
- 独特性：最优策略是否唯一？
- 随机性：最优策略是随机的还是确定性的？
- 算法：如何获得最优策略和最优状态值？

## 贝尔曼最优方程(Bellman optimality equation)

分析最优策略和最优状态价值的工具是贝尔曼最优方程（Bellman optimality equation, BOE）。通过求解此方程，我们可以获得最优策略和最优状态值。

接下来，我们介绍 BOE 的表达式，然后对其进行详细分析。

$$\begin{aligned}
v_\pi(s) =& \max_{\pi(s) \in \Pi(s)} \sum_{a \in \mathcal{A}} \pi(a|s) \left(\sum_{r \in \mathcal{R}} p(r|s,a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) v_\pi(s')\right) \\
=& \max_{\pi(s) \in \Pi(s)} \sum_{a \in \mathcal{A}} \pi(a|s) q_\pi(s,a)
\end{aligned}$$

这里 $\pi(s)$ 表示状态 $s$ 下的策略，$\Pi(s)$ 表示状态 $s$ 下所有可能策略的集合。

$v_\pi(s)$ 和 $v_\pi(s')$ 都是未知量，同时

$$ q_\pi(s,a) = \sum_{r \in \mathcal{R}} p(r|s,a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) v_\pi(s') $$

我们需要回答关于 BOE 的以下基本问题：
- 存在：这个方程有解吗？
- 独特性：该解决方案是否唯一？
- 算法：如何解这个方程？
- 最优性：解决方案如何与最优策略相关联？

一旦我们能够回答这些问题，我们将清楚地理解最优状态值和最优策略。