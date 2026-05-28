# 📚 OpenMOA Stream Wrapper 深度分析文档

> **文件路径**: `src/openmoa/stream/stream_wrapper.py`
> **版本**: OpenMOA 1.0.0
> **分析日期**: 2026-02-10

---

## 📋 目录

1. [概览与定位](#1-概览与定位)
2. [代码结构分解](#2-代码结构分解)
3. [逐行/逐块详解](#3-逐行逐块详解)
4. [数据流分析](#4-数据流分析)
5. [依赖关系](#5-依赖关系)
6. [特殊机制与技术点](#6-特殊机制与技术点)
7. [潜在问题与改进建议](#7-潜在问题与改进建议)
8. [使用示例](#8-使用示例)

---

## 1. 概览与定位

### 1.1 🎯 文件的主要目的和功能

`stream_wrapper.py` 是 OpenMOA 流处理框架的**核心装饰器模块（Decorator Module）**，提供了一套用于**模拟特征空间动态演化**的流包装器（Stream Wrappers）。

**核心功能**：
- 将**静态固定特征空间的数据集**转换为**特征空间随时间变化的动态数据流**
- 模拟真实在线学习场景中的**特征漂移（Feature Drift）**现象
- 为在线学习算法提供**标准化的特征演化基准测试环境**

### 1.2 🏗️ 在整个项目中的角色

```
┌─────────────────────────────────────────────────────────────────┐
│                     OpenMOA 架构层次图                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐                                            │
│  │   Datasets      │  ← 数据来源层 (UCI/MOA/自定义)              │
│  └────────┬────────┘                                            │
│           │                                                     │
│  ┌────────▼────────┐                                            │
│  │  Base Streams   │  ← 基础流层 (ARFF/CSV/LibSVM/Numpy)         │
│  │  (_stream.py)   │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│  ┌────────▼────────────────────────────────────────┐            │
│  │           Stream Wrappers  ⭐ 本文件               │            │
│  │  ┌──────────────┐ ┌─────────────────────────┐   │            │
│  │  │ShuffledStream│ │ OpenFeatureStream       │   │            │
│  │  └──────────────┘ │ TrapezoidalStream       │   │            │
│  │                   │ CapriciousStream        │   │            │
│  │                   │ EvolvableStream         │   │            │
│  │                   └─────────────────────────┘   │            │
│  └────────┬────────────────────────────────────────┘            │
│           │                                                     │
│  ┌────────▼────────┐                                            │
│  │  Learners       │  ← 学习器层 (FOBOS/ORF3V/OVFM/HOT...)      │
│  └────────┬────────┘                                            │
│           │                                                     │
│  ┌────────▼────────┐                                            │
│  │  Evaluation     │  ← 评估层 (Prequential/Holdout/Drift)      │
│  └─────────────────┘                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 🔧 解决的具体问题

| 问题 | 解决方案 |
|------|----------|
| **标签排序偏差 (Label Sorting Bias)** | `ShuffledStream` 打乱实例顺序 |
| **特征空间维度变化 (Varying Dimensionality)** | `OpenFeatureStream` 输出变长向量 + 全局索引映射 |
| **特征渐进出现 (Trapezoidal Data Stream)** | `TrapezoidalStream` / TDS 模式 |
| **特征随机缺失 (Capricious Data Stream)** | `CapriciousStream` / CDS 模式 |
| **特征分段演化 (Evolvable Data Stream)** | `EvolvableStream` / EDS 模式 |
| **索引偏移问题 (Index Shift Problem)** | `feature_indices` 元数据附加 |
| **NaN 兼容性 (Missing Value Compatibility)** | 固定维度 + NaN 填充方案 |

---

## 2. 代码结构分解

### 2.1 📦 类清单

| 类名 | 行号 | 继承自 | 职责 |
|------|------|--------|------|
| `OpenFeatureStream` | 7-288 | `Stream` | 变长输出的特征演化包装器（支持6种模式） |
| `TrapezoidalStream` | 290-457 | `Stream` | 固定维度 + NaN 填充的 TDS 模拟器 |
| `CapriciousStream` | 460-572 | `Stream` | 固定维度 + NaN 填充的 CDS 模拟器 |
| `EvolvableStream` | 576-772 | `Stream` | 固定维度 + NaN 填充的 EDS 模拟器 |
| `ShuffledStream` | 776-836 | `Stream` | 内存缓冲 + 随机打乱包装器 |

### 2.2 🔍 方法清单（按类分组）

#### OpenFeatureStream

| 方法名 | 行号 | 可见性 | 职责 |
|--------|------|--------|------|
| `__init__` | 30-101 | Public | 初始化参数、预计算时间表 |
| `_generate_dimension_schedule` | 103-114 | Private | 生成维度变化时间表（pyramid/inc/dec） |
| `_generate_feature_indices` | 116-132 | Private | 预计算每时刻的活跃特征索引 |
| `_generate_tds_offsets` | 134-157 | Private | 为每个特征分配"出生时间" |
| `_generate_eds_partitions` | 159-164 | Private | 将特征空间分割为 n 个分区 |
| `_calculate_eds_boundaries` | 166-186 | Private | 计算 EDS 各阶段的时间边界 |
| `_get_active_indices` | 188-236 | Private | **核心逻辑引擎**：返回当前活跃特征 ID |
| `next_instance` | 238-273 | Public | 返回下一个演化后的实例 |
| `has_more_instances` | 275-276 | Public | 检查是否还有更多实例 |
| `restart` | 278-280 | Public | 重置流状态 |
| `get_schema` | 282-284 | Public | 返回原始 Schema |
| `get_moa_stream` | 286-287 | Public | 返回 None（纯 Python 实现） |

#### TrapezoidalStream

| 方法名 | 行号 | 可见性 | 职责 |
|--------|------|--------|------|
| `__init__` | 310-359 | Public | 初始化、计算特征优先级排名 |
| `_generate_feature_ranking` | 361-378 | Private | 确定特征的"出生优先级" |
| `_generate_dimension_schedule` | 380-399 | Private | 计算每时刻的活跃特征数 k |
| `next_instance` | 401-441 | Public | 返回固定维度 + NaN 填充的实例 |
| `has_more_instances` | 443-447 | Public | 检查终止条件 |
| `restart` | 449-451 | Public | 重置流状态 |
| `get_schema` | 453-454 | Public | 返回 Schema |
| `get_moa_stream` | 456-457 | Public | 返回 None |

#### CapriciousStream

| 方法名 | 行号 | 可见性 | 职责 |
|--------|------|--------|------|
| `__init__` | 470-500 | Public | 初始化缺失率参数 |
| `_get_feature_mask` | 502-522 | Private | 伯努利试验生成缺失掩码 |
| `next_instance` | 524-556 | Public | 返回随机缺失的实例 |
| `has_more_instances` | 558-562 | Public | 检查终止条件 |
| `restart` | 564-566 | Public | 重置流状态 |
| `get_schema` | 568-569 | Public | 返回 Schema |
| `get_moa_stream` | 571-572 | Public | 返回 None |

#### EvolvableStream

| 方法名 | 行号 | 可见性 | 职责 |
|--------|------|--------|------|
| `__init__` | 594-636 | Public | 初始化分段参数 |
| `_generate_partitions` | 638-646 | Private | 顺序划分特征空间 |
| `_calculate_boundaries` | 648-676 | Private | 计算阶段边界 |
| `_get_active_mask` | 678-715 | Private | 返回布尔活跃掩码 |
| `next_instance` | 717-755 | Public | 返回阶段性演化的实例 |
| `has_more_instances` | 757-761 | Public | 检查终止条件 |
| `restart` | 763-765 | Public | 重置流状态 |
| `get_schema` | 767-768 | Public | 返回 Schema |
| `get_moa_stream` | 770-771 | Public | 返回 None |

#### ShuffledStream

| 方法名 | 行号 | 可见性 | 职责 |
|--------|------|--------|------|
| `__init__` | 784-806 | Public | 加载全部数据、生成打乱索引 |
| `has_more_instances` | 808-809 | Public | 检查指针位置 |
| `next_instance` | 811-820 | Public | 按打乱顺序返回实例 |
| `restart` | 822-826 | Public | 重置指针 |
| `get_schema` | 828-829 | Public | 返回 Schema |
| `get_num_instances` | 831-833 | Public | 返回总实例数 |
| `get_moa_stream` | 835-836 | Public | 返回 None |

### 2.3 🔗 组件调用关系图

```
外部调用方 (demo/benchmark)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                     ShuffledStream                         │
│   ┌─────────────────────────────────────────────────────┐ │
│   │ __init__:                                            │ │
│   │   while base.has_more_instances():                   │ │
│   │       _instances.append(base.next_instance())        │ │
│   │   _indices = shuffle(range(n))                       │ │
│   └─────────────────────────────────────────────────────┘ │
│   ┌─────────────────────────────────────────────────────┐ │
│   │ next_instance:                                       │ │
│   │   return _instances[_indices[_ptr++]]                │ │
│   └─────────────────────────────────────────────────────┘ │
└───────────────────────┬───────────────────────────────────┘
                        │ (作为 base_stream 传入)
                        ▼
┌───────────────────────────────────────────────────────────┐
│                   OpenFeatureStream                        │
│   ┌─────────────────────────────────────────────────────┐ │
│   │ __init__:                                            │ │
│   │   根据 evolution_pattern 调用对应的预计算方法:         │ │
│   │   ├── pyramid/inc/dec → _generate_dimension_schedule │ │
│   │   │                   → _generate_feature_indices    │ │
│   │   ├── tds → _generate_tds_offsets                    │ │
│   │   └── eds → _generate_eds_partitions                 │ │
│   │           → _calculate_eds_boundaries                │ │
│   └─────────────────────────────────────────────────────┘ │
│   ┌─────────────────────────────────────────────────────┐ │
│   │ next_instance:                                       │ │
│   │   1. base_stream.next_instance()                     │ │
│   │   2. _get_active_indices(t) ← 根据模式分发           │ │
│   │   3. x_subset = x_full[active_indices]               │ │
│   │   4. LabeledInstance.from_array(schema, x_subset, y) │ │
│   │   5. instance.feature_indices = active_indices       │ │
│   │   6. return instance                                 │ │
│   └─────────────────────────────────────────────────────┘ │
└───────────────────────┬───────────────────────────────────┘
                        │
                        ▼
                   FOBOSClassifier
                        │
                        ▼
               _get_sparse_x(instance)
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
  feature_indices?   x_index/x_value?   else
        │               │               │
        ▼               ▼               ▼
 return (indices,   return (x_index,  filter(x!=0 & !nan)
         inst.x)         x_value)
```

---

## 3. 逐行/逐块详解

### 3.1 🔧 导入与类型声明 (Lines 1-6)

```python
import numpy as np                                    # 数值计算核心库
from typing import Literal, Optional, List            # 类型注解
from openmoa.stream import Stream                     # 流基类
from openmoa.stream._stream import Schema             # 流结构描述
from openmoa.instance import LabeledInstance, RegressionInstance  # 实例类型
```

**设计意图**：
- 使用 `Literal` 类型限制参数取值范围，IDE 可提供自动补全
- 导入 `Stream` 基类确保所有包装器遵循统一接口

---

### 3.2 📐 OpenFeatureStream 详解

#### 3.2.1 构造函数 (Lines 30-101)

```python
def __init__(
    self,
    base_stream: Stream,                              # 原始数据流
    d_min: int = 2,                                   # 最小活跃特征数
    d_max: Optional[int] = None,                      # 最大特征数（默认=原始维度）
    evolution_pattern: Literal["pyramid", ...] = "pyramid",  # 演化模式
    total_instances: int = 10000,                     # 流总长度（用于计算时间表）
    feature_selection: Literal["prefix", ...] = "prefix",    # 特征选择策略
    missing_ratio: float = 0.0,                       # CDS 缺失率
    random_seed: int = 42,                            # 随机种子
    tds_mode: Literal["random", "ordered"] = "random",       # TDS 模式
    n_segments: int = 2,                              # EDS 分段数
    overlap_ratio: float = 1.0,                       # EDS 重叠比例
):
```

**参数取值范围**：

| 参数 | 类型 | 取值范围 | 默认值 | 说明 |
|------|------|----------|--------|------|
| `d_min` | int | [1, d_max] | 2 | 最少保留的活跃特征数 |
| `d_max` | int | [d_min, 原始维度] | None(=原始维度) | 特征空间上界 |
| `evolution_pattern` | str | pyramid/incremental/decremental/tds/cds/eds | "pyramid" | 演化策略 |
| `total_instances` | int | > 0 | 10000 | 流长度，用于计算时间轴 |
| `feature_selection` | str | prefix/suffix/random | "prefix" | 确定性模式的特征选取方式 |
| `missing_ratio` | float | [0.0, 1.0) | 0.0 | CDS 模式的特征缺失概率 |
| `tds_mode` | str | random/ordered | "random" | TDS 的特征出生顺序 |
| `n_segments` | int | >= 2 | 2 | EDS 的分区数 |
| `overlap_ratio` | float | >= 0 | 1.0 | EDS 重叠期与稳定期的比值 |

**关键初始化逻辑**：

```python
# 验证 d_max 不超过原始维度
original_d = base_stream.get_schema().get_num_attributes()
self.d_max = d_max if d_max is not None else original_d
if self.d_max > original_d:
    raise ValueError(f"d_max ({self.d_max}) cannot exceed original feature count ({original_d})")

# 创建可复现的随机数生成器
self._rng = np.random.RandomState(random_seed)
self._current_t = 0  # 时间步计数器

# 根据模式预计算
if evolution_pattern in ["pyramid", "incremental", "decremental"]:
    self._dimension_schedule = self._generate_dimension_schedule()    # 维度时间表
    self._feature_indices_cache = self._generate_feature_indices()    # 索引缓存
elif evolution_pattern == "tds":
    self._feature_offsets = self._generate_tds_offsets()              # 特征出生时间
elif evolution_pattern == "eds":
    self._eds_partitions = self._generate_eds_partitions()            # 特征分区
    self._eds_boundaries = self._calculate_eds_boundaries()           # 阶段边界
# cds 模式是随机的，无需预计算
```

> 💡 **设计模式：策略模式 (Strategy Pattern)**
> 不同演化模式通过 `evolution_pattern` 参数选择，初始化和运行时逻辑都通过条件分支分发到对应策略。

---

#### 3.2.2 维度时间表生成 (Lines 103-114)

```python
def _generate_dimension_schedule(self) -> np.ndarray:
    """生成每个时间步的目标维度数"""
    dims = np.zeros(self.total_instances, dtype=int)

    if self.evolution_pattern == "pyramid":
        half = self.total_instances // 2
        dims[:half] = np.linspace(self.d_min, self.d_max, half)   # 前半段：上升
        dims[half:] = np.linspace(self.d_max, self.d_min, self.total_instances - half)  # 后半段：下降

    elif self.evolution_pattern == "incremental":
        dims = np.linspace(self.d_min, self.d_max, self.total_instances)  # 单调递增

    elif self.evolution_pattern == "decremental":
        dims = np.linspace(self.d_max, self.d_min, self.total_instances)  # 单调递减

    return dims.astype(int)
```

**可视化示例**（total_instances=1000, d_min=2, d_max=10）：

```
Pyramid:          Incremental:      Decremental:
    ^                   ^                 ^
10 -|    /\          10-|        ___   10-|___
    |   /  \            |       /         |   \
    |  /    \           |      /          |    \
 2 -|_/      \_       2-|_____/         2-|     \___
    +----------→        +--------→        +--------→
    0   500  1000       0      1000       0      1000
```

---

#### 3.2.3 特征索引预计算 (Lines 116-132)

```python
def _generate_feature_indices(self) -> list:
    """为每个时间步预计算具体的特征索引"""
    indices_list = []
    for t in range(self.total_instances):
        d_current = self._dimension_schedule[t]     # 当前时刻的目标维度

        if self.feature_selection == "prefix":
            indices = np.arange(d_current)           # 选择前 d_current 个特征
            # 例：d_current=3 → [0, 1, 2]

        elif self.feature_selection == "suffix":
            indices = np.arange(self.d_max - d_current, self.d_max)  # 选择后 d_current 个
            # 例：d_max=10, d_current=3 → [7, 8, 9]

        elif self.feature_selection == "random":
            rng_t = np.random.RandomState(self.random_seed + t)  # 时间步相关的随机种子
            indices = rng_t.choice(self.d_max, d_current, replace=False)
            indices.sort()
            # 例：d_max=10, d_current=3 → [2, 5, 8] (随机选择但排序后输出)

        indices_list.append(indices)
    return indices_list
```

> 💡 **可复现性技巧**：`random` 模式使用 `random_seed + t` 作为种子，确保同一时间步在不同运行中产生相同结果。

---

#### 3.2.4 TDS 偏移量生成 (Lines 134-157)

```python
def _generate_tds_offsets(self) -> np.ndarray:
    """为每个特征分配"出生时间"(birth time)"""
    offsets = np.zeros(self.d_max, dtype=int)
    time_step = self.total_instances // 10   # 将时间线分为10个阶段

    if self.tds_mode == "random":
        # 随机打乱特征顺序，均匀分配到10个阶段
        indices = self._rng.permutation(self.d_max)
        for i in range(self.d_max):
            offsets[indices[i]] = (i % 10) * time_step
        # 例：d_max=30, indices=[5,12,0,...]
        #     Feature 5 → bucket 0 → offset 0
        #     Feature 12 → bucket 1 → offset 100
        #     Feature 0 → bucket 2 → offset 200

    elif self.tds_mode == "ordered":
        # 按特征索引顺序分配（特征0先出生，特征N-1最后出生）
        for i in range(self.d_max):
            bucket = (i * 10) // self.d_max   # 将索引映射到0-9
            offsets[i] = bucket * time_step
        # 例：d_max=30
        #     Feature 0-2 → bucket 0 → offset 0
        #     Feature 3-5 → bucket 1 → offset 100
        #     ...

    return offsets
```

**TDS Ordered 模式可视化**（d_max=10, total_instances=1000）：

```
特征索引:    0  1  2  3  4  5  6  7  8  9
出生时间:    0  0  0  100 100 200 200 300 300 400
            |________|________|________|________|
              Bucket0  Bucket1  Bucket2  Bucket3

时间线演示:
t=0:   活跃特征 = [0,1,2]                    (3个)
t=150: 活跃特征 = [0,1,2,3,4]                (5个)
t=350: 活跃特征 = [0,1,2,3,4,5,6,7]          (8个)
t=500: 活跃特征 = [0,1,2,3,4,5,6,7,8,9]      (全部10个)
```

---

#### 3.2.5 EDS 分区与边界计算 (Lines 159-186)

```python
def _generate_eds_partitions(self) -> List[np.ndarray]:
    """将特征空间顺序划分为 n 个不相交的子集"""
    all_indices = np.arange(self.d_max)
    partitions = np.array_split(all_indices, self.n_segments)
    return [np.sort(p) for p in partitions]
    # 例：d_max=10, n_segments=3
    #     → [[0,1,2,3], [4,5,6], [7,8,9]]

def _calculate_eds_boundaries(self) -> List[int]:
    """计算 EDS 各阶段的时间边界"""
    n = self.n_segments
    r = self.overlap_ratio

    # 公式推导：
    # 总时间 = n个稳定期 + (n-1)个重叠期
    # 总时间 = n*L + (n-1)*r*L = L * (n + r*(n-1))
    # 解出 L = total_instances / (n + r*(n-1))
    L = self.total_instances / (n + r * (n - 1))

    boundaries = []
    current_pos = 0.0
    total_stages = 2 * n - 1   # 稳定和重叠交替

    for i in range(total_stages):
        if i % 2 == 0:  # 偶数索引 = 稳定期
            current_pos += L
        else:           # 奇数索引 = 重叠期
            current_pos += L * r
        boundaries.append(int(current_pos))

    boundaries[-1] = self.total_instances  # 确保最后一个边界精确
    return boundaries
```

**EDS 演化图示**（n_segments=3, overlap_ratio=1.0, total=1000）：

```
L = 1000 / (3 + 1.0*(3-1)) = 1000 / 5 = 200

阶段边界: [200, 400, 600, 800, 1000]

Stage 0 (t=0~199):    P0 活跃       [0,1,2,3]
Stage 1 (t=200~399):  P0 ∪ P1      [0,1,2,3,4,5,6]      ← 重叠期
Stage 2 (t=400~599):  P1 活跃       [4,5,6]
Stage 3 (t=600~799):  P1 ∪ P2      [4,5,6,7,8,9]        ← 重叠期
Stage 4 (t=800~999):  P2 活跃       [7,8,9]

可视化:
        ┌──────┐
    P0  │██████│
        └──────┘
             ┌──────────────┐
    P0∪P1    │              │
             └──────────────┘
                  ┌──────┐
    P1            │██████│
                  └──────┘
                       ┌──────────────┐
    P1∪P2              │              │
                       └──────────────┘
                            ┌──────┐
    P2                      │██████│
                            └──────┘
        +------+------+------+------+------→ t
        0     200    400    600    800   1000
```

---

#### 3.2.6 核心逻辑引擎：_get_active_indices (Lines 188-236)

```python
def _get_active_indices(self, t: int) -> np.ndarray:
    """
    🔥 这是整个类的决策核心
    输入: 当前时间步 t
    输出: 当前应该活跃的全局特征ID数组
    """

    # 1️⃣ 确定性模式（pyramid/incremental/decremental）
    if self.evolution_pattern in ["pyramid", "incremental", "decremental"]:
        if t < len(self._feature_indices_cache):
            return self._feature_indices_cache[t]  # 直接查表
        return np.arange(self.d_min)  # 越界保护

    # 2️⃣ TDS 模式
    elif self.evolution_pattern == "tds":
        # 返回所有 offset <= t 的特征
        return np.where(self._feature_offsets <= t)[0]

    # 3️⃣ CDS 模式（随机缺失）
    elif self.evolution_pattern == "cds":
        rng_t = np.random.RandomState(self.random_seed + t)
        # 伯努利试验：以 (1-missing_ratio) 的概率保留特征
        mask = rng_t.rand(self.d_max) > self.missing_ratio
        indices = np.where(mask)[0]
        if len(indices) == 0:
            indices = np.array([0])  # 保底：至少保留一个特征
        return indices

    # 4️⃣ EDS 模式（分段演化）
    elif self.evolution_pattern == "eds":
        # 找到当前所属的阶段
        stage_idx = 0
        for i, boundary in enumerate(self._eds_boundaries):
            if t < boundary:
                stage_idx = i
                break
        else:
            stage_idx = len(self._eds_boundaries) - 1

        if stage_idx % 2 == 0:
            # 偶数阶段：稳定期，单个分区活跃
            partition_idx = stage_idx // 2
            return self._eds_partitions[partition_idx]
        else:
            # 奇数阶段：重叠期，相邻两个分区都活跃
            prev = (stage_idx - 1) // 2
            indices = np.concatenate([
                self._eds_partitions[prev],
                self._eds_partitions[prev + 1]
            ])
            indices.sort()
            return indices

    return np.arange(self.d_max)  # 默认：所有特征活跃
```

> 🔑 **关键洞察**：这个方法是纯函数（Pure Function），对于相同的 `t` 和初始化参数，总是返回相同的结果。这使得整个流是**可复现的（Reproducible）**。

---

#### 3.2.7 实例生成核心：next_instance (Lines 238-273)

```python
def next_instance(self):
    """返回下一个特征演化后的实例"""
    if not self.has_more_instances():
        return None

    # Step 1: 从底层流获取原始实例
    base_instance = self.base_stream.next_instance()
    if base_instance is None:
        return None

    # Step 2: 确定当前活跃的全局特征 ID
    active_indices = self._get_active_indices(self._current_t)

    # Step 3: 从完整特征向量中切片
    x_full = np.array(base_instance.x)
    # 防御性编程：确保索引不越界
    valid_indices = active_indices[active_indices < len(x_full)]
    if len(valid_indices) == 0:
        valid_indices = np.array([0])

    x_subset = x_full[valid_indices]  # 物理截短

    # Step 4: 构建新的 Instance 对象
    if self._schema.is_classification():
        new_instance = LabeledInstance.from_array(
            self._schema, x_subset, base_instance.y_index
        )
    else:
        new_instance = RegressionInstance.from_array(
            self._schema, x_subset, base_instance.y_value
        )

    # Step 5: 🔥 附加全局特征索引（解决 Index Shift 问题）
    new_instance.feature_indices = valid_indices

    self._current_t += 1
    return new_instance
```

**数据变换可视化**：

```
输入 (来自 base_stream):
┌─────────────────────────────────────────────┐
│ x_full = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]    │  (6维)
│ y_index = 1                                 │
└─────────────────────────────────────────────┘
                    │
                    ▼ _get_active_indices(t=100)
                    │ 返回 [1, 3, 5] (假设 TDS 模式)
                    │
┌─────────────────────────────────────────────┐
│ valid_indices = [1, 3, 5]                   │
│ x_subset = x_full[[1,3,5]] = [0.2, 0.4, 0.6]│  (3维)
└─────────────────────────────────────────────┘
                    │
                    ▼ 构建新实例
                    │
输出 (传给 Learner):
┌─────────────────────────────────────────────┐
│ new_instance.x = [0.2, 0.4, 0.6]           │  (物理3维)
│ new_instance.y_index = 1                    │
│ new_instance.feature_indices = [1, 3, 5]    │  ← 全局ID元数据
└─────────────────────────────────────────────┘
```

> 🔑 **Index Shift 问题解决方案**：
> 虽然物理向量是 `[0.2, 0.4, 0.6]`（位置0,1,2），但 `feature_indices=[1,3,5]` 告诉学习器这些值对应的是**全局特征1、3、5**。学习器更新权重时会写入 `W[[1,3,5]]` 而非 `W[[0,1,2]]`。

---

### 3.3 📐 TrapezoidalStream 详解

与 `OpenFeatureStream` 的核心区别在 `next_instance` 方法：

```python
def next_instance(self):
    # ...
    # 1. 获取当前活跃特征数 k
    t_idx = min(self._current_t, self.total_instances - 1)
    num_active = self._dimension_schedule[t_idx]

    # 2. 确定哪些特征活跃（基于优先级排名）
    active_indices = self._feature_ranking[:num_active]

    # 3. 创建 NaN 画布（固定维度）
    x_full = np.full(self.d_max, np.nan)  # 🔥 关键：预填充 NaN

    # 4. 只填入活跃位置的值
    x_base = np.array(base_instance.x)[:self.d_max]
    x_full[active_indices] = x_base[active_indices]

    # 5. 构建实例（维度始终 = d_max）
    modified_instance = LabeledInstance.from_array(
        self._schema, x_full, base_instance.y_index
    )
    # 注意：这里没有 feature_indices 元数据！
    # ...
```

**对比表**：

| 特性 | OpenFeatureStream | TrapezoidalStream |
|------|-------------------|-------------------|
| 输出向量维度 | 可变（= 活跃特征数） | 固定（= d_max） |
| 不活跃特征表示 | 不存在于向量中 | `np.nan` 占位 |
| `feature_indices` 元数据 | ✅ 有 | ❌ 无 |
| 适用算法 | 稀疏感知（FOBOS, FTRL） | NaN 处理（OVFM, OSLMF） |
| 内存效率 | 更高（短向量） | 较低（全长向量） |

---

### 3.4 📐 ShuffledStream 详解

```python
class ShuffledStream(Stream):
    def __init__(self, base_stream: Stream, random_seed: int = 42):
        # 1️⃣ 立即加载全部数据到内存
        self._instances = []
        while base_stream.has_more_instances():
            try:
                inst = base_stream.next_instance()
                if inst is not None:
                    self._instances.append(inst)
            except Exception:
                break

        self.n_instances = len(self._instances)

        # 2️⃣ 创建打乱的索引映射
        self._indices = np.arange(self.n_instances)
        self._rng = np.random.RandomState(random_seed)
        self._rng.shuffle(self._indices)

        self._ptr = 0  # 当前读取指针

    def next_instance(self):
        if not self.has_more_instances():
            return None
        # 按打乱后的索引访问
        real_idx = self._indices[self._ptr]
        instance = self._instances[real_idx]
        self._ptr += 1
        return instance
```

**内存模型**：

```
原始流:     [inst_0, inst_1, inst_2, inst_3, inst_4, inst_5]
                │
                ▼ 全部加载
内存列表:   [inst_0, inst_1, inst_2, inst_3, inst_4, inst_5]
                │
                ▼ 生成索引并打乱
索引数组:   [3, 0, 5, 1, 4, 2]
                │
                ▼ 按打乱索引访问
输出顺序:   inst_3 → inst_0 → inst_5 → inst_1 → inst_4 → inst_2
```

> ⚠️ **警告**：此类会将整个数据集加载到内存，仅适用于 UCI 量级（MB）数据集。对于 GB 级大数据流，这会导致内存溢出。

---

## 4. 数据流分析

### 4.1 完整数据流图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              数据流全景图                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐                                                       │
│  │   磁盘文件        │                                                       │
│  │ .arff/.csv/.libsvm│                                                       │
│  └────────┬─────────┘                                                       │
│           │ (读取)                                                           │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │  openmoa.datasets │                                                       │
│  │  例: Magic04()    │  返回 Stream 对象                                     │
│  └────────┬─────────┘  Schema: {d=10, classes=2}                            │
│           │            instance.x = [v0, v1, ..., v9] (固定10维)            │
│           │                                                                 │
│  ┌────────▼─────────────────────────────────────────────────────────────┐   │
│  │                        ShuffledStream                                 │   │
│  │  输入: base_stream (有序)                                             │   │
│  │  处理: 全量加载 → 打乱索引                                             │   │
│  │  输出: 打乱后的 stream                                                 │   │
│  │  维度: 不变 (10维)                                                     │   │
│  └────────┬─────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           │ (作为 base_stream 传入)                                          │
│           │                                                                 │
│  ┌────────▼─────────────────────────────────────────────────────────────┐   │
│  │                     OpenFeatureStream (TDS)                           │   │
│  │  输入: shuffled_stream                                                │   │
│  │  处理:                                                                │   │
│  │    t=0:   active=[0,1]        → x=[v0,v1]        feature_indices=[0,1]│   │
│  │    t=100: active=[0,1,2,3]    → x=[v0,v1,v2,v3]  feature_indices=[0,1,2,3]│
│  │    t=500: active=[0,1,...,9]  → x=完整10维        feature_indices=[0..9]│  │
│  │  输出: 变长向量 + feature_indices 元数据                               │   │
│  └────────┬─────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           │ next_instance() 返回                                            │
│           │                                                                 │
│  ┌────────▼─────────────────────────────────────────────────────────────┐   │
│  │                     FOBOSClassifier                                   │   │
│  │  接收: instance                                                       │   │
│  │                                                                       │   │
│  │  _get_sparse_x(instance):                                            │   │
│  │    检测 hasattr(instance, "feature_indices") → True                  │   │
│  │    返回 (instance.feature_indices, instance.x)                       │   │
│  │          即 ([0,1,2,3], [v0,v1,v2,v3])                               │   │
│  │                                                                       │   │
│  │  train():                                                             │   │
│  │    W[[0,1,2,3], 0] -= eta * grad * [v0,v1,v2,v3]                     │   │
│  │    (稀疏更新：只修改涉及的4行权重)                                      │   │
│  │                                                                       │   │
│  │  权重矩阵: W.shape = (10, 1)   ← 全局维度，不随 t 变化                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 数据变换追踪表

以 TDS 模式处理一个具体实例为例：

| 阶段 | 位置 | 数据形态 | 维度 |
|------|------|----------|------|
| 原始 | 磁盘 `.libsvm` | `1 1:0.3 2:0.5 ... 10:0.9` | 10 |
| 加载 | `LibsvmStream` | `instance.x = [0.3, 0.5, ..., 0.9]` | 10 |
| 打乱 | `ShuffledStream` | 同上（仅顺序改变） | 10 |
| TDS t=50 | `OpenFeatureStream` | `x=[0.3,0.5], feature_indices=[0,1]` | 2 |
| TDS t=300 | `OpenFeatureStream` | `x=[0.3,0.5,0.6,0.7,0.8], fi=[0,1,2,3,4]` | 5 |
| TDS t=900 | `OpenFeatureStream` | `x=[完整10维], fi=[0,1,...,9]` | 10 |
| 学习器 | `FOBOSClassifier` | `indices, values = _get_sparse_x(inst)` | 变量 |

---

## 5. 依赖关系

### 5.1 外部库依赖

| 库 | 版本要求 | 用途 |
|-----|----------|------|
| `numpy` | >= 1.20 | 数值计算、数组操作、随机数生成 |
| `typing` | 内置 (Python 3.9+) | 类型注解 (`Literal`, `Optional`, `List`) |

### 5.2 内部模块依赖

```
stream_wrapper.py
    │
    ├── openmoa.stream.Stream        (基类)
    │   └── _stream.py:296
    │
    ├── openmoa.stream._stream.Schema (流结构描述)
    │   └── _stream.py:49
    │
    └── openmoa.instance
        ├── LabeledInstance          (分类实例)
        │   └── instance.py:176
        └── RegressionInstance       (回归实例)
            └── instance.py:230
```

### 5.3 被依赖关系

```
stream_wrapper.py
    │
    ├── openmoa.stream.__init__.py   (导出接口)
    │   └── 导出: OpenFeatureStream, TrapezoidalStream,
    │            CapriciousStream, EvolvableStream, ShuffledStream
    │
    └── demo/*.py                    (17个benchmark脚本)
        ├── demo_fobos_benchmark_binary.py
        ├── demo_ovfm_benchmark_binary.py
        └── ...
```

---

## 6. 特殊机制与技术点

### 6.1 🎨 设计模式

#### 6.1.1 装饰器模式 (Decorator Pattern)

```python
# 所有包装器都接受一个 base_stream 参数
class OpenFeatureStream(Stream):
    def __init__(self, base_stream: Stream, ...):
        self.base_stream = base_stream

    def next_instance(self):
        base_instance = self.base_stream.next_instance()  # 委托给底层
        # ... 进行增强处理 ...
        return enhanced_instance
```

**优点**：
- 包装器可以任意嵌套：`OpenFeatureStream(ShuffledStream(Magic04()))`
- 遵循开闭原则：扩展新的演化模式不需要修改现有代码

#### 6.1.2 策略模式 (Strategy Pattern)

```python
# evolution_pattern 参数选择不同的算法策略
if evolution_pattern == "tds":
    self._feature_offsets = self._generate_tds_offsets()
elif evolution_pattern == "eds":
    self._eds_partitions = self._generate_eds_partitions()
# ...
```

#### 6.1.3 模板方法模式 (Template Method Pattern)

所有包装器共享相同的方法签名（继承自 `Stream` 基类）：

```python
class Stream(ABC):
    @abstractmethod
    def next_instance(self) -> Instance: ...

    @abstractmethod
    def has_more_instances(self) -> bool: ...

    def get_schema(self) -> Schema: ...

    def restart(self): ...
```

### 6.2 🔐 可复现性机制

```python
# 全局随机状态
self._rng = np.random.RandomState(random_seed)

# 时间步相关的局部随机状态（用于 CDS 模式）
rng_t = np.random.RandomState(self.random_seed + t)
mask = rng_t.rand(self.d_max) > self.missing_ratio
```

> 💡 使用 `np.random.RandomState` 而非全局 `np.random`，避免外部代码的随机调用影响可复现性。

### 6.3 🛡️ 防御性编程

```python
# 1. 参数边界检查
if self.d_max > original_d:
    raise ValueError(f"d_max ({self.d_max}) cannot exceed original feature count ({original_d})")

if n_segments < 2:
    raise ValueError("n_segments must be >= 2 for EDS pattern")

# 2. 空结果保护
indices = np.where(mask)[0]
if len(indices) == 0:
    indices = np.array([0])  # 保底：至少返回一个特征

# 3. 越界保护
valid_indices = active_indices[active_indices < len(x_full)]
if len(valid_indices) == 0:
    valid_indices = np.array([0])

# 4. 安全的时间索引访问
t_idx = min(self._current_t, self.total_instances - 1)
```

### 6.4 ⚡ 性能优化

#### 6.4.1 预计算 (Pre-computation)

```python
# 初始化时预计算所有时间步的索引，避免运行时重复计算
self._dimension_schedule = self._generate_dimension_schedule()
self._feature_indices_cache = self._generate_feature_indices()
```

**时间复杂度对比**：

| 方法 | 初始化 | 每次 next_instance |
|------|--------|-------------------|
| 预计算 | O(T) | O(1) 查表 |
| 动态计算 | O(1) | O(d_max) |

对于长流（T >> d_max），预计算更优。

#### 6.4.2 NumPy 向量化

```python
# 向量化布尔索引，而非 Python 循环
mask = rng_t.rand(self.d_max) > self.missing_ratio
indices = np.where(mask)[0]

# 向量化切片
x_subset = x_full[valid_indices]
```

### 6.5 🔧 动态属性附加

```python
# 为实例动态添加 feature_indices 属性
new_instance.feature_indices = valid_indices
```

这是 Python 的动态特性，无需修改 `Instance` 类定义即可扩展其功能。下游代码通过 `hasattr` 检测：

```python
if hasattr(instance, "feature_indices"):
    return instance.feature_indices, instance.x
```

---

## 7. 潜在问题与改进建议

### 7.1 🐛 已识别问题

#### 问题 1：ShuffledStream 内存消耗

```python
# 当前实现：全量加载
while base_stream.has_more_instances():
    self._instances.append(base_stream.next_instance())
```

**风险**：对于大数据集（如 GB 级），会导致内存溢出。

**建议改进**：
```python
# 方案 1：使用内存映射文件 (Memory-Mapped File)
# 方案 2：分块洗牌 (Chunk-based Shuffling)
# 方案 3：使用 Reservoir Sampling 进行流式洗牌
```

#### 问题 2：CDS 模式的极端情况

```python
mask = rng_t.rand(self.d_max) > self.missing_ratio
indices = np.where(mask)[0]
if len(indices) == 0:
    indices = np.array([0])  # 只保留特征 0
```

**风险**：当 `missing_ratio` 接近 1.0 时，频繁触发保底机制，特征 0 被过度强调。

**建议改进**：
```python
if len(indices) == 0:
    # 随机选择一个特征，而非总是选 0
    indices = np.array([rng_t.randint(0, self.d_max)])
```

#### 问题 3：Schema 不一致

```python
def get_schema(self) -> Schema:
    return self._schema  # 返回原始 Schema（d_max 维）
```

**风险**：对于 `OpenFeatureStream`，返回的 Schema 描述的是 d_max 维空间，但实际输出的向量可能只有 d_min 维。某些依赖 Schema 的下游代码可能产生维度不匹配错误。

**建议改进**：
```python
# 方案 1：文档明确说明行为
# 方案 2：创建动态 Schema 或返回 None 表示可变维度
```

### 7.2 📈 性能优化建议

#### 建议 1：惰性计算 (Lazy Evaluation)

```python
# 当前：初始化时预计算所有 total_instances 个索引
# 改进：使用生成器按需计算
def _generate_feature_indices(self):
    for t in range(self.total_instances):
        yield self._compute_indices_for_t(t)
```

#### 建议 2：缓存优化

```python
# 对于 EDS 模式，分区信息可以用元组存储以节省内存
self._eds_partitions = tuple(np.sort(p) for p in partitions)
```

### 7.3 🔍 可读性改进

#### 建议 1：提取魔法数字

```python
# 当前
time_step = self.total_instances // 10  # 为什么是 10？

# 改进
TDS_NUM_BUCKETS = 10  # TDS 模式的阶段数
time_step = self.total_instances // TDS_NUM_BUCKETS
```

#### 建议 2：统一方法命名

```python
# 当前混用
_get_active_indices()   # OpenFeatureStream
_get_active_mask()      # EvolvableStream
_get_feature_mask()     # CapriciousStream

# 建议统一为
_compute_active_features() → (indices | mask)
```

---

## 8. 使用示例

### 8.1 🚀 快速上手

```python
from openmoa.datasets import Magic04
from openmoa.stream import ShuffledStream, OpenFeatureStream
from openmoa.classifier._fobos_classifier import FOBOSClassifier

# 1. 加载数据集并打乱
base = ShuffledStream(Magic04(), random_seed=42)
n_total = base.get_num_instances()  # 19020

# 2. 包装为 TDS 演化流
stream = OpenFeatureStream(
    base_stream=base,
    evolution_pattern="tds",
    tds_mode="ordered",
    d_min=2,
    total_instances=n_total,
    random_seed=42
)

# 3. 初始化学习器
schema = stream.get_schema()
learner = FOBOSClassifier(
    schema=schema,
    alpha=1.0,
    lambda_=0.0001,
    regularization="l1"
)

# 4. 在线学习循环
correct = 0
total = 0
while stream.has_more_instances():
    instance = stream.next_instance()

    # 观察特征演化
    print(f"t={total}: dim={len(instance.x)}, indices={instance.feature_indices}")

    # Test-Then-Train
    pred = learner.predict(instance)
    if pred == instance.y_index:
        correct += 1
    learner.train(instance)
    total += 1

print(f"Accuracy: {correct/total:.4f}")
```

### 8.2 🔄 CDS 模式示例（随机缺失）

```python
stream = OpenFeatureStream(
    base_stream=ShuffledStream(Magic04()),
    evolution_pattern="cds",
    missing_ratio=0.4,  # 40% 的特征缺失
    total_instances=19020,
    random_seed=42
)

# 观察缺失模式
for i, inst in enumerate(stream):
    if i >= 5:
        break
    print(f"t={i}: present={inst.feature_indices}, missing={set(range(10)) - set(inst.feature_indices)}")
```

输出示例：
```
t=0: present=[0,2,4,5,7,9], missing={1,3,6,8}
t=1: present=[1,2,3,6,8,9], missing={0,4,5,7}
t=2: present=[0,1,3,5,6,7,8], missing={2,4,9}
...
```

### 8.3 📊 EDS 模式示例（分段演化）

```python
stream = OpenFeatureStream(
    base_stream=ShuffledStream(Magic04()),
    evolution_pattern="eds",
    n_segments=3,      # 3个分区
    overlap_ratio=0.5, # 重叠期 = 0.5 * 稳定期
    total_instances=1000,
    random_seed=42
)

# 追踪阶段转换
stages = {}
for i, inst in enumerate(stream):
    key = tuple(inst.feature_indices)
    if key not in stages:
        stages[key] = i
        print(f"t={i}: New stage, features={key}")
```

### 8.4 🔀 与 NaN 方案对比

```python
from openmoa.stream import TrapezoidalStream

# 固定维度 + NaN 方案
stream_nan = TrapezoidalStream(
    base_stream=ShuffledStream(Magic04()),
    evolution_mode="ordered",
    total_instances=1000
)

inst = stream_nan.next_instance()
print(f"Vector: {inst.x}")        # 完整10维，部分位置是 NaN
print(f"Has NaN: {np.isnan(inst.x).sum()}")  # 统计 NaN 数量
print(f"Has feature_indices: {hasattr(inst, 'feature_indices')}")  # False
```

---

## 📎 附录

### A. 术语对照表

| 中文术语 | 英文术语 | 缩写 |
|----------|----------|------|
| 数据流 | Data Stream | Stream |
| 特征空间 | Feature Space | - |
| 特征漂移 | Feature Drift | - |
| 概念漂移 | Concept Drift | - |
| 梯形数据流 | Trapezoidal Data Stream | TDS |
| 随机缺失数据流 | Capricious Data Stream | CDS |
| 分段演化数据流 | Evolvable Data Stream | EDS |
| 索引偏移问题 | Index Shift Problem | - |
| 先测后训 | Test-Then-Train / Prequential | - |
| 装饰器模式 | Decorator Pattern | - |
| 稀疏感知 | Sparse-aware | - |

### B. 相关文件快速索引

| 文件 | 路径 | 说明 |
|------|------|------|
| Stream 基类 | `src/openmoa/stream/_stream.py:296` | 抽象基类定义 |
| Schema 类 | `src/openmoa/stream/_stream.py:49` | 流结构描述 |
| Instance 类 | `src/openmoa/instance.py:38` | 数据实例基类 |
| FOBOS 分类器 | `src/openmoa/classifier/_fobos_classifier.py` | 稀疏感知学习器示例 |
| 包导出 | `src/openmoa/stream/__init__.py` | 公开接口 |
| Benchmark 示例 | `demo/demo_fobos_benchmark_binary.py` | 完整使用示例 |

---

> 📝 **文档版本**: 1.0
> **最后更新**: 2026-02-10
> **作者**: Claude Code Analysis
