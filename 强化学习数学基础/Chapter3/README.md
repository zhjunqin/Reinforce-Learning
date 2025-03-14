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
v(s) =& \max_{\pi(s) \in \Pi(s)} \sum_{a \in \mathcal{A}} \pi(a|s) \left(\sum_{r \in \mathcal{R}} p(r|s,a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) v(s')\right) \\
=& \max_{\pi(s) \in \Pi(s)} \sum_{a \in \mathcal{A}} \pi(a|s) q(s,a)
\end{aligned}$$

这里 $\pi(s)$ 表示状态 $s$ 下的策略，$\Pi(s)$ 表示状态 $s$ 下所有可能策略的集合。

$v(s)$ 和 $v(s')$ 都是未知量，同时

$$ q(s,a) = \sum_{r \in \mathcal{R}} p(r|s,a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) v(s') $$

我们需要回答关于 BOE 的以下基本问题：
- 存在：这个方程有解吗？
- 独特性：该解决方案是否唯一？
- 算法：如何解这个方程？
- 最优性：解决方案如何与最优策略相关联？

一旦我们能够回答这些问题，我们将清楚地理解最优状态值和最优策略。

### 最大化 BOE 的右边

如何计算右边的最大化的值，通过一个示例来表示：

示例：给定 $q_1, q_2, q_3 \in \mathbb{R}$ ，找到如下等式最大值时的 $c_1,c_2,c_3$：

$$ \sum^3_{i=1} c_i q_i = c_1 q_1 + c_2 q_2 + c_3 q_3  $$

其中 $c_1 + c_2 + c_3 = 1$ 且  $c_1, c_2, c_3 > 0$。

不失一般性，假设 $ q_3 \geq q_1,q_2 $，则最优解为 $c^*_3=1$ 且 $c^*_1=c^*_2=0$，这是由于：

$$ q_3 = (c_1+c_2+c_3)q_3 = c_1 q_3 + c_2 q_3 + c_3 q_3 \geq c_1 q_1 + c_2 q_2 + c_3 q_3, \quad \forall c_1,c_2,c_3 $$

由此我们可以类推，由于 $ \sum_a \pi(a|s) = 1$，则可以得到：


$$ \sum_{a \in \mathcal{A}} \pi(a|s) q_\pi(s,a) \leq \sum_{a \in \mathcal{A}} \pi(a|s) \max_{a \in \mathcal{A}} q_\pi(s,a) = \max_{a \in \mathcal{A}} q_\pi(s,a) $$

当 $\pi(a|s)$ 满足如下条件时，上述等式成立。

$$
\pi(a|s) = 
\begin{cases}
1, & a = a^*, \\
0, & a \neq a^*.
\end{cases}$$

这里的 $a^* = \argmax_a q(s,a)$。

可以理解为，求解得到的最优策略为在状态 $s$ 下选择能够让动作价值 $q_\pi(s,a)$ 最大的动作。

$$v(s) = \max_{\pi(s) \in \Pi(s)} \sum_{a \in \mathcal{A}} \pi(a|s) q(s,a)$$

### BOE 的矩阵-向量形式

BOE 的矩阵-向量形式为：

$$ v = \max_{\pi \in \Pi}( r_\pi + \gamma P_\pi v) $$

其中：

$$\begin{aligned}
v &\in \mathbb{R}^{|\mathcal{S}|} \\
[r_\pi]_s &= \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{r \in \mathcal{R}} p(r|s,a)r \\
[P_\pi]_{s,s'} &= p_\pi(s'|s) = \sum_{a \in \mathcal{A}} \pi(a|s)  p(s'|s,a)
\end{aligned}$$

由于最优策略 $\pi$ 是由 $v$ 来决定，因此，右边可以写成 $v$ 的函数，表示为：

$$ f(v) = \max_{\pi \in \Pi}(r_\pi + \gamma P_\pi v）$$

这样，BOE 就可以用简洁的形式表示为：

$$ v = f(v)$$

### 压缩映射定理（Contraction mapping theorem）

**压缩映射定理（Contraction mapping theorem）**，也被称之为 **Banach 不动点定理**。

为了介绍**压缩映射定理**，展开介绍如下几个基本概念：

#### **度量空间**
在数学中，度量空间（Metric space）是具有距离这一个概念的集合，具体来说，是装配了一个称为度量的函数，用以表示此集合中任两个成员间的距离。

度量空间中最符合人们对于现实直观理解的为三维欧几里得空间。事实上，“度量”的概念即是欧几里得距离四个周知的性质之推广。欧几里得度量定义了两点间之距离为连接这两点的直线段之长度。定义如下：

![](./assets/chapter3_metric_space.png)

#### **压缩映射**
设 $(X,d)$ 为非空的度量空间，对于映射 $T: X \rightarrow X$，如果存在 $q \in (0,1)$ 使得 

$$d(T(x), T(y)) < q \cdot d(x,y), \forall x,y \in X $$

那么映射 $T$ 称之为压缩映射。从直观上理解，一个压缩映射 $T$，使得 $T(x)$ 和 $T(y)$ 之间的距离比 $x$ 和 $y$ 之间的距离近。

#### **压缩映射定理（巴拿赫不动点定理）**
一个映射 $T: X \rightarrow X$ 是一个压缩映射，则 $T$ 在 $X$ 内**有且只有**一个不动点 $x*$，也就是：

$$ T(x^*) = x^*$$

而且这个不动点可以用如下迭代的方式求出：

从 $X$ 内的任意一个元素 $x_0$ 开始，定义一个迭代序列 $x_n = T(x_{n-1})$，其中 $n=1,2,3,...$，那么这个序列收敛，并且收敛的极限为 $x^*$。以下的不等式描述了收敛的速度：

$$ d(x^*, x_n) \leq \frac{q^n}{1-q} d(x_1, x_0) $$

等价地：

$$ d(x^*, x_{n+1}) \leq \frac{q}{1-q} d(x_{n+1}, x_{n}) $$

且

$$ d(x^*, x_{n+1}) \leq q \cdot d(x^*, x_{n}) $$

满足以上不等式的最小的 $q$ 有时称为**利普希茨常数**。
这个序列也是柯西序列。在数学中，**柯西序列（Cauchy sequence）**，也称为基本列，是指一个元素随着序数的增加而愈发靠近的数列。






## 参考文献
- https://zh.wikipedia.org/wiki/%E5%8E%8B%E7%BC%A9%E6%98%A0%E5%B0%84
- https://zh.wikipedia.org/wiki/%E5%BA%A6%E9%87%8F%E7%A9%BA%E9%97%B4
- https://zhuanlan.zhihu.com/p/494932745
- https://zh.wikipedia.org/wiki/%E6%9F%AF%E8%A5%BF%E5%BA%8F%E5%88%97