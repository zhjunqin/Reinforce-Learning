# 价值迭代和策略迭代

本章将介绍三个算法，分别是价值迭代（value iteration）算法，策略迭代（policy
iteration）算法和截断策略迭代（ truncated policy iteration）算法。

本章介绍的算法称为动态规划（dynamic programming）算法，需要系统模型。

## 价值迭代（value iteration）

### 矩阵向量的形式

算法是通过压缩映射定理的迭代的方式来求解贝尔曼最优方程：

$$v_{k+1} = \max_{\pi \in \Pi}(r_\pi + \gamma P_\pi v_k), \quad k=0,1,2,...$$

对于任意给定的 $v_0$，当 $k \rightarrow \infin $ 时，$v_k$ 和 $\pi_k$ 分别收敛到最优状态价值和最优策略。

算法在每次迭代时都有两个步骤。

#### 1. 策略更新（policy update）

目标是找到可以解决以下优化问题的策略：

$$\pi_{k+1} = \argmax_{\pi}(r_\pi + \gamma P_\pi v_k)$$

这里的 $v_k$ 是在上一次迭代中获得的。特别的，$v_0$ 是初始化时的值。

#### 2. 价值更新（value update）

通过如下方式计算新的 $v_{k+1}$：


$$v_{k+1} = r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_k$$

计算出来的 $v_{k+1}$ 被用到下一次迭代中。

### 逐个元素的形式

考虑步骤 $k$ 和状态 $s$：

#### 1. 策略更新步骤（policy update step）

逐个元素形式的策略更新步骤为：

$$ \pi_{k+1}(s) = \argmax_{\pi(s)} \sum_{a \in \mathcal{A}} \pi(a|s) \left(\sum_{r \in \mathcal{R}} p(r|s,a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) v_k(s')\right), \forall s \in S$$

解决上述最优化问题的最优策略为：

$$
\pi_{k+1}(a|s) = 
\begin{cases}
1, & a = a^*_k(s), \\
0, & a \neq a^*_k(s).
\end{cases}$$

这里的 $a^*_k(s) = \argmax_a q_k(s,a)$。如果 $\argmax_a q_k(s,a)$ 有多个解的话，可以从中选择任意一个。这样选择出来的策略称之为为贪心策略。

#### 2. 价值更新步骤（value update step）

逐个元素形式的价值更新步骤为：

$$ v_{k+1}(s) = \sum_{a \in \mathcal{A}} \pi_{k+1}(a|s) \left(\sum_{r \in \mathcal{R}} p(r|s,a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) v_k(s')\right), \forall s \in S$$

替换 $\pi_{k+1}(a|s)$ 为上一个步骤得到的贪心策略，则得到：

$$ v_{k+1}(s) = \max_a q_k(s,a)$$

### 算法步骤

上述的步骤可以描述为：

$$ v_k(s) \rightarrow q_k(s,a) \rightarrow new\ greedy\ policy\ \pi_{k+1}(s) \rightarrow new\ value\ v_{k+1}(s) = \max_aq_k(s,a)$$

具体算法步骤如下图

![](./assets/chapter4_value_iteration.png)

### 示例说明

使用如下 $2 \times 2$ 的网格世界来做示例。

目标是 $s4$，奖励 $r_{boundary} = r_{forbidden} = -1$， $r_{target} = 1$，其他 $r_{other} = 0$。折扣因子 $\gamma = 0.9$。

![](./assets/chapter4_value_iteration_example.png)

#### 步骤 $k=0$

不失一般性，设置初始状态价值 $v_0(s_1) = v_0(s_2)= v_0(s_3) = v_0(s_4) =0$。

- 计算动作价值 $q_0(s,a)$：

  ![](./assets/chapter4_value_iteration_example_1.png)

  将 $v_0(s_1) = v_0(s_2)= v_0(s_3) = v_0(s_4) =0$ 代入得到：

  ![](./assets/chapter4_value_iteration_example_2.png)

- 策略更新
  策略 $\pi_1$ 是对每个状态 $s$ 选择最大 $q(s,a)$ 的动作，因此得到

  $$ \pi_1(a_5|s_1) = 1, \pi_1(a_3|s_2) = 1, \pi_1(a_2|s_3) = 1, \pi_1(a_5|s_4) = 1 $$

  其中 $\pi_1(a_5|s_1)$ 是随机选择的，假定这么选。

- 价值更新
  价值 $v_1$ 的获得是通过对每一个状态 $s$ 的状态价值更新为最大的 $q(s,a)$。因此得到：

  $$ v_1(s_1) = 0,\ v_1(s_2) = 1,\ v_1(s_3) = 1,\ v_1(s_4) = 1 $$

#### 步骤 $k=1$

- 计算动作价值 $q_1(s,a)$：
  
  将上个步骤得到的 $v_1$ 代入到上面的 q-table 表 4.1 中可以得到：

  ![](./assets/chapter4_value_iteration_example_3.png)

- 策略更新
  策略 $\pi_2$ 是对每个状态 $s$ 选择最大 $q(s,a)$ 的动作，因此得到

  $$ \pi_1(a_3|s_1) = 1,\ \pi_1(a_3|s_2) = 1,\ \pi_1(a_2|s_3) = 1,\ \pi_1(a_5|s_4) = 1 $$

- 价值更新
  价值 $v_2$ 的获得是通过对每一个状态 $s$ 的状态价值更新为最大的 $q(s,a)$。因此得到：

  $$ v_2(s_1) = \gamma 1,\ v_2(s_2) = 1+\gamma 1,\ v_2(s_3) = 1+\gamma1,\ v_2(s_4) = 1+\gamma 1 $$

可以看到策略 $\pi_2$ 已经是最优策略。