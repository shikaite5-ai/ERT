# 三个项目的证据边界
| 项目 | 代码/资料支持的事实 | 对选题的含义 |
|---|---|---|
| [CRAFT](https://proceedings.mlsys.org/paper_files/paper/2026/hash/3a7f9e485845dac27423375c934cb4db-Abstract-Conference.html) | 公开 artifact 做 8 个真实 DeepSeek-R1/Kimi-K2 traces 的 CPU replay；先对 batch 取平均，再做逐层分配 | 适合快速做风险、预算和跨 workload 实验 |
| [DeepSeek V3 / EPLB](https://arxiv.org/abs/2412.19437) | 按估计负载做 hierarchical/global 副本与 placement；负载预测明确不在仓库范围内 | 静态计划很依赖预测值，仍给 uncertainty-aware 研究留下空间 |
| [MoonEP](https://github.com/moonshotAI/moonep) | 在线动态冗余、训练梯度回收、零拷贝通信；每 rank 有 `epn` 个 prefetch slots | 证明动态复制在训练通信上可行，但它不是 serving/KV 预算系统 |

# CRAFT
## 1. CRAFT
- 针对问题：expert 热度通常高度不均衡；即使把 expert 尽量均匀地放到 GPU 上，少数热门 expert 仍可能让某些 GPU 成为瓶颈。
- 核心思想：**设计专家副本数量的分配算法**。不是给每一层都复制同样多的 expert，而是根据每个 MoE layer 的 routing trace 和复制带来的收益，细粒度、分层地决定复制数量；同时把复制带来的显存成本纳入预算。
- 算法处理问题：
<1> 每一个 MoE layer 要复制多少 expert slot；
<2> 复制出来的 physical expert 如何跨 GPU 分布；
<3> 如何在固定 GPU/显存预算下，让最坏 GPU 的负载尽量接近平均负载。
- 算法细节：
### STEP1: 重放 routing trace
副本分配器的输入是离线记录的 **expert-load trace**<span style="color: orange;">（推理过程中就记录下来的）</span>。可以把它理解为一个张量：对每个 inference batch、每个 layer、每个 logical expert，记录该 expert 收到的 token/负载。

先统计整份 trace 中各 logical expert 的**总负载**，并根据候选副本数安排各 GPU 的 **physical expert slot**<span style="color: orange;">（某个 GPU 在某一 MoE 层上拥有的一个 physical expert 位置或名额）</span>；再结合负载统计和 slot 布局，确定 **logical-to-physical 映射**<span style="color: orange;">（专家复制后，会有多个 physical expert 实例执行同一个 logical expert 的工作）</span>；

随后逐个 batch **重放**记录的 expert 负载：把 trace 中记录的每一批负载，按照当前候选复制与放置方案重新汇总到 GPU 上，然后计算这批流量的 balancedness。

$$
\text{balancedness} = \frac{\text{平均 GPU load}}{\text{最大 GPU load}}
$$

### STEP2: 逐层估计 replication benefit
统计在不同候选副本规模下，每个 MoE layer 的 balancedness 能比“只做 placement、不复制”改善多少。

**这一步的核心产物是逐层、逐候选副本规模的收益表**，即得到每一层的 $B_{l,q}$、相对 placement-only 的原始增益 $\Delta_{l,q}=B_{l,q}-B_{l,0}$，以及归一化收益 $u_{l,q}$。只有保留这张表，<span style="color: orange;">后续才能判断“哪一层值得复制、值得复制多少”</span>。

候选规模为 $q=0,4,8,16,32$，其中 q=0 是 placement-only baseline。对每个候选 $q$，代码执行一遍完整重放，对每个batch计算：

$$
B_{b,l,q}=
\frac{\text{该 batch 各 GPU 负载的平均值}}
     {\text{该 batch 最忙 GPU 的负载}}
$$

对所有 batch 求平均，得到:

$$
B_{l,q}=\frac{1}{N}\sum_b B_{b,l,q}
$$

扣除 placement-only 的基线，并除以q（归一化）：

$$
u_{l,q}=\frac{B_{l,q}-B_{l,0}}{q}
$$

$u_{l,q}$ 表示“平均每增加一个 replica slot 获得多少 balancedness 增益”。除以 $q$ 很重要，因为增加 32 个 slot 显然比增加 4 个更容易提高 balancedness，但不应因此直接认为 32 一定更值得。

随后代码计算所有层、所有非零候选的全局平均值：

$$
\bar u=
\frac{1}{L_M(K-1)}
\sum_{l}\sum_{q>0}u_{l,q}
$$

其中 $L_M$ 是 MoE 层数，$K-1$ 是非零候选数。这个 $\bar u$ 被用作公开实现中的相对收益阈值。

因此，$\bar u$ 不是 STEP2 最终要获得的唯一答案，也不是被最大化的目标。它只是把逐层收益表压缩成一个全局平均尺度，用于判断某个候选的每-slot 收益相对全局平均有多高；真正决定后续分配的是完整的逐层收益表。

### STEP3: 在显存/副本预算下选择 layer-wise allocation
STEP2 已得到：
$$
\Delta_{l,q}=B_{l,q}-B_{l,0}
$$
表示第 $l$ 层采用 $q$ 个额外副本时，相对于 placement-only 的 balancedness 增益。

设 $x_{l,q}\in\{0,1\}$：是否给第 $l$ 层选择 $q$ 个额外副本。

每层只能选择一个候选方案：
$$
\sum_q x_{l,q}=1
$$
总副本数不能超过预算：
$$
\sum_l\sum_q q\,x_{l,q}\le r
$$
目标是最大化全部层的收益：
$$
\max \sum_l\sum_q \Delta_{l,q}x_{l,q}
$$
这就是 **MCKP**：每个 layer 是一个“组”，每组必须从若干**候选副本数**<span style="color: orange;">（论文设计为0，4，8，16，32，因为复制收益通常逐步饱和，2 的幂足以刻画主要趋势）</span>中选择一个；所有层的选择共同消耗同一个副本预算。

例如，<span style="color: orange;">不能因为第 20 层和第 51 层各自单独看都适合复制 16 份，就给两层都分配 16 份</span>；必须看总预算能否容纳，并把副本让给“每单位副本收益更高”的层。因此，STEP3 解决的是层间资源竞争。

这个最优分配方案的获取，论文使用动态规划求解。若 $F_l(b)$ 表示处理完前 $l$ 层、已经使用 $b$ 个副本时的最大总收益，则：

$$
F_l(b)=
\max_{q\le b}
\left[
F_{l-1}(b-q)+\Delta_{l,q}
\right]
$$

所有层计算完成后，找到不超过预算 $r$ 的最优 $F_{L_M}(b)$，再回溯得到每层的选择 $q_l$。

最终结果：<span style="color: orange;">每个 MoE 层最终拥有多少个物理专家槽位（**原有logical expert数 + 新增副本数**）</span>

$$
(q_{l_1},q_{l_2},\ldots,q_{l_M})
$$

### STEP4: 跨 GPU 分配 replica slots
已经获得每个 MoE 层的 $q_l$ 后，要将这个**容量分配**给这一层 $g$ 张 GPU：

先按整除结果分配 base 数量，再把余数交错放到不同 GPU，避免余数总落在相邻 GPU 或同一局部位置。

### STEP5: capacity-aware greedy placement
还要决定**具体哪个 logical expert 的哪个 replica 放到哪个 GPU**。CRAFT 的公开代码沿用/实现了一套 EPLB 风格的 packing：优先把负载较高的 expert 放到当前较空闲、且满足容量/节点约束的 GPU。

最终需要输出：
- log2phy[l,e]：logical expert $e$ 对应哪些 physical expert；
- phy2log[l,p]：physical expert $p$ 属于哪个 logical expert；
- logcnt[l,e]：logical expert $e$ 一共有几份 physical 实例；
- 每份 physical 实例所在的 GPU 和 GPU 内槽位编号。

<span style="color: orange;">具体逻辑很简单，就是贪心算法：</span>

#### 第一项计算：哪些 expert 获得新增副本
代码每次选择当前“每份实例平均负载”最大的 logical expert。

#### 第二项计算：把 physical expert 放到 GPU
接下来对每一层：
1. 将 physical expert 按权重 \(w_{l,p}\) 从大到小排序；
2. 先放负载大的 physical expert；
3. 对每个 physical expert，排除已经装满的 GPU；
4. 尽量排除已经在本节点拥有同一个 logical expert 副本的 GPU；<span style="color: orange;">（否则该 expert 输入时流量会特别大）</span>
5. 如果节点约束无法满足，则退化为避免同一 GPU；
6. 在合法候选中选择放置代价最小的 GPU。

代码实现上，就用了一个1000：1的系数：
$$
\text{score}(g)=1000\cdot H_{l,g}+A_{l,g}
$$
分数低为佳。其中：
- $H_{l,g}$：当前已经放入 GPU $g$ 的累计物理 expert 负载；
- $A_{l,g}$：GPU $g$ 在该层的槽位容量。

这样就可以表达主要因素为选择当前累计负载最低的 GPU，次要因素为槽位容量。

# Deepseek-V3

- **把副本放置与 group、node、EP 阶段和 shared expert 绑定到实际 serving 拓扑**

| 设计 | 具体解释 | 为什么有效 |
|---|---|---|
| （1）先 group-to-node，再 node 内复制，再 GPU packing | 先以 **expert group**（logical expert的分组） 为整体，<span style="color: orange;">把负载较高的 group 优先放到较空闲的节点</span>；然后<span style="color: orange;">在该节点内部增加热门 expert 的副本</span>，最后把 physical experts 分配到节点内的各张 GPU | 保留 expert group 的<span style="color: orange;">节点局部性，减少跨节点 all-to-all</span> |
| （2）Prefill 与 Decode 使用不同 EP 和冗余规模 | Prefill 使用较小的 EP32，并给每张 GPU 增加约一个冗余槽位；Decode 使用 EP320，每张 GPU 约只承载一个 expert，并把较多槽位用于冗余和 shared experts<span style="color: orange;">（**EP size**：把 256 个 routed experts 分散到多少张 GPU 上；**冗余**：在已有 physical experts 之外，再增加多少个热门 expert 副本；**Prefill**：输入 token batch 大，**加载一个 expert 权重的成本可以由大量 token 的计算摊薄**，反之如果 EP 设大，Prefill 每层处理大量 token，跨节点通信开销更容易成为主要问题；**Decode**：每步 token 很少，但需要大量并发，因此使用 EP320，**让每 GPU 平均只承载一个 expert**）</span> | 两个阶段瓶颈不同：Prefill 偏计算，Decode 偏内存访问和通信延迟 |
| （3）Decode 把 <span style="color: orange;">shared expert</span> 当作 routed expert 计入负载 | Decode 时不再把 shared expert 当作单独的特殊路径，而是把它看成一个每 token 必选的 routed expert，因此每 token 实际选择 9 个 experts 参与负载统计和放置。<span style="color: orange;">这带来三个效果：1.shared expert 进入 top-k 负载统计；2.shared expert 可以像热门 routed expert 一样拥有副本；3.它会被纳入 GPU placement 和 all-to-all 规划，而不是被当作不计入负载的特殊旁路。</span> | Shared expert 每 token 必选，是天然热点；遗漏它会低估负载 |
| （4）周期约 10 分钟重新识别热点并调整冗余 | serving 框架持续采样 post-routing<span style="color: orange;">（路由之后的阶段）</span> 的 expert 负载，每隔一个统计窗口重新选出热门 experts，并更新副本集合和放置方案；窗口示例约为 10 分钟 | 跟踪 workload drift，同时避免每 step 迁移权重的开销 |
| （5）Node-limited routing 最多 4 个节点 | router 不只按 expert 分数选择，还要求每个 token 最多被发送到 4 个节点；节点通常按该节点上高分 expert 的**聚合 affinity**<span style="color: orange;">（**affinity**：router 对“某个 token 与某个 expert 匹配程度”的分数；**聚合affinity**：把一个节点上所有 experts 的分数整理后，取其中最高的若干项再相加，得到一个“这个节点对当前 token 的整体吸引力”）</span> 来选择 | <span style="color: orange;">限制跨节点流量</span>，使计算通信重叠更可行 |
| （6）无辅助损失 routing bias | 给每个 expert 维护一个 **routing bias**<span style="color: orange;">（一个参数）</span>；top-k 用“原始 affinity + bias”来选 expert，但**最终 gate 权重**<span style="color: orange;">（router 选出 top-k experts 后，真正乘到每个 expert 输出上的那个归一化系数）</span>仍使用原始 affinity。批次中<span style="color: orange;">过载 expert 的 bias 降低（**更难被选中**），欠载 expert 的 bias 提高（**更容易被选中**）</span> | 在 batch 层均衡负载，同时保留专家领域专门化，避免辅助损失损害模型质量 |

对（6）的解释：

对 token $t$ 和 routed expert $i$，先计算：
$$
s_{i,t}
=
\sigma(u_t^{\top}e_i)
$$
其中：
- $u_t$：FFN 输入；
- $e_i$：expert \(i\) 的 centroid 向量；
- $\sigma$：sigmoid；
- $s_{i,t}$：这个 token 对 expert \(i\) 的 affinity。

假设 top-k 选中的 expert 集合是：
$$
\mathcal S_t
$$

则最终 gate 权重：
$$
g_{i,t}
=
\begin{cases}
\dfrac{s_{i,t}}
{\sum_{j\in\mathcal S_t}s_{j,t}},
& i\in\mathcal S_t\\
0,&\text{otherwise}
\end{cases}
$$

# MoonEP
- 目标：让每个 EP GPU 进程在每层都收到相同的**真实 token 事件数**。

**真实 token 事件数** ≠ **token 数**，因为一个 token 可以选择多个专家，如果多个专家在一个 GPU 上，就算多个事件。

## STEP1: 固定 Buffer
```python
buffer = Buffer(
    S=4096,
    H=7168,
    K=8,
    E=256,
    num_ep_ranks=8,
    num_sms=32,
    token_padding=128,
)
```

MoonEP 的关键选择是**固定形状**<span style="color: orange;">（固定 buffer 容量和 kernel 可用的 tensor shape，否则每次都要知道 routing 后的实际行数才能分配 tensor，那么dispatch/combine/kernel异步启动参数等许多参数都需要再确定）</span>。

## STEP2: 在线 Planning
根据当前 batch 的 router 结果，在线生成一张“把哪些 token-expert 事件送到哪个 rank、需要<span style="color: orange;">预取（prefetch）哪些远程 expert（即制作专家副本）</span>”的通信计划。

planning 输入：

- 每个 source rank 的 `tokens_per_expert`；
- 每个 routed-token 事件的 top-k expert id；
- `NvS`、`token_padding`、rank 与 expert 布局。

planning 输出 [`MoonEPCommPlan`](MoonEP-master/moonep/planning.py#L32)：

| 字段 | 含义 |
|---|---|
| `dst` | 每个 routed-token 事件的目标 rank 与目标 buffer offset；负值表示同 token 的重复目标编码 |
| `experts_to_copy` | 形状 `[R, epn]`，每个目标 rank 需要从远程复制的 expert id，`-1` 表示空槽 |
| `zero_fill_ranges` | padded segment 中需要清零的 token 区间 |
| `remote_stats` | remote expert 数量等统计 |
| `dup_groups` / `dup_loffs` / `dup_counts` | 同一 token 多个路由事件落到同一 rank 时的 dedup 结构 |

### 目标：每个 rank 的负载均衡

设计每个 rank 的真实token事件**容量**为 $N$，**实际负载**为 $G_h$，那么可以定义 rank(h) 的 group 的不均衡量：

$$
b_h=G_h-N
$$

- \(b_h>0\)：该 home group 超载，需要向外迁移 token；
- \(b_h<0\)：该 rank 空闲，可以接收 remote expert 的 token；
- \(b_h=0\)：已经平衡。

### 迁移策略是：

找到当前最超载的 group (h)；  
找到当前最空闲的 destination rank (d)；  
把一个完整的空闲配额从 (h) 移到 (d)；  
该 destination 被填满后，不再从另一个 source group 接收 token。

## STEP3: prefetch（迁移），dispatch
这里有一个 **zero-copy** 技术：

### 普通非 zero-copy：

**假设** dispatch 后，token hidden 已经位于目标 rank 的 NVLink 通信 buffer：
```
NVLink communication buffer
        ↓ 额外 copy
普通用户 tensor
        ↓
expert FFN 读取
        ↓
expert 输出写入普通 tensor
        ↓ 额外 copy
communication buffer
        ↓
combine
```
缺点是：
- 多一次 buffer 到 expert 输入的 copy；
- 多一次 expert 输出到 buffer 的 copy；
- 多占一份 buffer；
- 增加 HBM 访问和 kernel 启动；
- 多一次同步或 stream 协调。

### MoonEP zero-copy：

dispatch 返回的不是新 tensor，而是 communication buffer 的 **view**（类似于指针）：
```
NVLink communication buffer
        ↓
hidden_nvsh = communication buffer view
        ↓
expert FFN 直接读取
        ↓
expert 输出原地写回同一个 buffer
        ↓
combine 直接消费这个 buffer
```


## STEP4: 反向传播
设最终输出梯度为：
$$
\nabla \mathbf z_A
$$
反向依次经过：
1. 依据保存的 **plan**，反向把输出 gradient 送回前向的 expert 输出位置
2. 每个 expert 做 FFN backward
3. 把 K 份 hidden gradient 汇总（**来自K个 expert**）回原 token
4. 把远程 expert 副本的权重梯度送回 owner rank
