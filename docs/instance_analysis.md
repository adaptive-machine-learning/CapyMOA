# 📚 OpenMOA Instance 模块深度分析文档

> **文件路径**: `src/openmoa/instance.py`
> **版本**: OpenMOA 1.0.0
> **分析日期**: 2026-02-11

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

`instance.py` 是 OpenMOA 框架的**数据实例（Data Instance）核心模块**，定义了数据流中单个数据点的抽象表示。它是连接**数据源（Stream）** 与 **学习器（Learner）** 之间的桥梁。

**核心功能**：
- 定义数据实例的统一抽象（特征向量 + 目标变量）
- 提供 **Python ↔ Java (MOA)** 双向互操作能力
- 支持分类任务（`LabeledInstance`）和回归任务（`RegressionInstance`）
- 实现惰性求值（Lazy Evaluation）优化性能

### 1.2 🏗️ 在整个项目中的角色

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        OpenMOA 数据流架构                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────┐                                                   │
│   │  Data Sources   │  磁盘文件 / MOA生成器 / NumPy数组                  │
│   └────────┬────────┘                                                   │
│            │                                                            │
│   ┌────────▼────────┐                                                   │
│   │     Stream      │  数据流抽象层                                      │
│   │   (_stream.py)  │  - ARFFStream, CSVStream, NumpyStream             │
│   └────────┬────────┘                                                   │
│            │ next_instance()                                            │
│            │                                                            │
│   ┌────────▼────────────────────────────────────────────────────────┐   │
│   │                    Instance  ⭐ 本文件                            │   │
│   │  ┌──────────────────────────────────────────────────────────┐   │   │
│   │  │  Instance (基类)                                          │   │   │
│   │  │    ├── .x           → 特征向量 (NumPy ndarray)            │   │   │
│   │  │    ├── .schema      → 流结构描述                          │   │   │
│   │  │    └── .java_instance → MOA Java 对象                     │   │   │
│   │  ├──────────────────────────────────────────────────────────┤   │   │
│   │  │  LabeledInstance (分类)                                   │   │   │
│   │  │    ├── .y_index     → 类别索引 (int: 0, 1, 2, ...)       │   │   │
│   │  │    └── .y_label     → 类别标签 (str: "yes", "no", ...)   │   │   │
│   │  ├──────────────────────────────────────────────────────────┤   │   │
│   │  │  RegressionInstance (回归)                                │   │   │
│   │  │    └── .y_value     → 目标值 (float: 17.949, ...)        │   │   │
│   │  └──────────────────────────────────────────────────────────┘   │   │
│   └────────┬────────────────────────────────────────────────────────┘   │
│            │                                                            │
│   ┌────────▼────────┐     ┌────────────────┐                            │
│   │  Python Learner │     │  MOA Learner   │                            │
│   │  (FOBOS, FTRL)  │     │  (HOT, NB)     │                            │
│   │  使用 .x, .y_*  │     │  使用 .java_*  │                            │
│   └─────────────────┘     └────────────────┘                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 🔧 解决的具体问题

| 问题 | 解决方案 |
|------|----------|
| **Python/Java 互操作** | 双向惰性转换：Python ndarray ↔ MOA InstanceExample |
| **任务类型差异** | 继承体系：Instance → LabeledInstance / RegressionInstance |
| **性能优化** | 惰性求值：仅在访问时才进行转换 |
| **类型安全** | 类型别名（FeatureVector, LabelIndex, TargetValue）+ 类型注解 |
| **循环导入** | `TYPE_CHECKING` 条件导入 |

---

## 2. 代码结构分解

### 2.1 📦 组件清单

#### 函数

| 函数名 | 行号 | 可见性 | 职责 |
|--------|------|--------|------|
| `_features_to_string` | 17-35 | Private | 格式化特征向量用于 `__repr__` |

#### 类

| 类名 | 行号 | 继承自 | 职责 |
|------|------|--------|------|
| `Instance` | 38-163 | - | 基类：无标签的数据实例 |
| `LabeledInstance` | 166-270 | `Instance` | 分类实例：带类别标签 |
| `RegressionInstance` | 273-365 | `Instance` | 回归实例：带连续目标值 |

### 2.2 🔍 方法清单（按类分组）

#### Instance（基类）

| 方法名 | 行号 | 类型 | 职责 |
|--------|------|------|------|
| `__init__` | 47-69 | Constructor | 初始化实例（支持 Java/NumPy 两种输入） |
| `from_java_instance` | 71-75 | @classmethod | 工厂方法：从 Java 对象创建 |
| `from_array` | 77-106 | @classmethod | 工厂方法：从 NumPy 数组创建 |
| `schema` | 108-111 | @property | 返回流结构描述 |
| `x` | 113-125 | @property | **惰性求值**：返回特征向量 |
| `_set_y` | 127-133 | Protected | 设置目标变量（子类覆盖） |
| `java_instance` | 135-155 | @property | **惰性求值**：返回 MOA Java 对象 |
| `__repr__` | 157-163 | Magic | 字符串表示 |

#### LabeledInstance

| 方法名 | 行号 | 类型 | 职责 |
|--------|------|------|------|
| `__init__` | 189-197 | Constructor | 初始化分类实例 |
| `from_array` | 199-237 | @classmethod | 工厂方法：从数组+标签索引创建 |
| `y_label` | 239-242 | @property | 返回类别标签字符串 |
| `y_index` | 244-256 | @property | **惰性求值**：返回类别索引 |
| `_set_y` | 258-260 | Protected | 覆盖：设置分类标签 |
| `__repr__` | 262-270 | Magic | 字符串表示（含标签信息） |

#### RegressionInstance

| 方法名 | 行号 | 类型 | 职责 |
|--------|------|------|------|
| `__init__` | 293-301 | Constructor | 初始化回归实例 |
| `from_array` | 303-340 | @classmethod | 工厂方法：从数组+目标值创建 |
| `y_value` | 342-351 | @property | **惰性求值**：返回目标值 |
| `__repr__` | 353-360 | Magic | 字符串表示（含目标值） |
| `_set_y` | 362-364 | Protected | 覆盖：设置回归目标值 |

### 2.3 🔗 类继承与调用关系

```
                    ┌─────────────────┐
                    │    Instance     │  基类
                    │                 │
                    │ + schema        │
                    │ + x             │  惰性：_x ↔ _java_instance
                    │ + java_instance │
                    │ + _set_y()      │  模板方法（子类覆盖）
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                                 │
   ┌────────▼────────┐             ┌──────────▼────────┐
   │ LabeledInstance │             │ RegressionInstance│
   │                 │             │                   │
   │ + y_index       │             │ + y_value         │
   │ + y_label       │             │                   │
   │ + _set_y() 覆盖 │             │ + _set_y() 覆盖   │
   └─────────────────┘             └───────────────────┘
```

**调用链示例**：

```
Stream.next_instance()
    │
    ├── 情况1：来自 MOA Java 流
    │   └── LabeledInstance(schema, java_instance)
    │       └── Instance.__init__(schema, java_instance)
    │           └── self._java_instance = java_instance
    │
    ├── 情况2：来自 NumpyStream
    │   └── LabeledInstance.from_array(schema, x, y_index)
    │       └── LabeledInstance(schema, (x, y_index))
    │           └── Instance.__init__(schema, x)
    │               └── self._x = x
    │
    └── 后续访问:
        ├── instance.x (首次访问)
        │   └── if _x is None: 从 _java_instance 提取
        │
        └── instance.java_instance (首次访问)
            └── if _java_instance is None: 从 _x 构建
```

---

## 3. 逐行/逐块详解

### 3.1 🔧 导入与类型声明 (Lines 1-14)

```python
from typing import TYPE_CHECKING                    # 条件导入标记

import numpy as np                                  # 数值计算
from com.yahoo.labs.samoa.instances import DenseInstance  # MOA 密集实例类
from moa.core import InstanceExample                # MOA 实例包装器
from typing import Optional, Union, Tuple           # 类型注解
from jpype import JArray, JDouble                   # Java 数组桥接

from openmoa.type_alias import FeatureVector, Label, LabelIndex, TargetValue

# 🔑 关键技术：避免循环导入
if TYPE_CHECKING:
    from openmoa.stream import Schema
```

**`TYPE_CHECKING` 详解**：

```python
# 问题：instance.py 需要 Schema 类型，Schema 定义在 _stream.py
#       _stream.py 又需要 Instance 类型
#       直接导入会形成循环依赖

# 解决方案：
if TYPE_CHECKING:  # 仅在类型检查器（如 mypy）运行时为 True
    from openmoa.stream import Schema  # 运行时不执行此导入

# 使用时用字符串引用（Forward Reference）
def __init__(self, schema: "Schema", ...):  # "Schema" 而非 Schema
```

---

### 3.2 📐 辅助函数 (Lines 17-35)

```python
def _features_to_string(
    x: np.ndarray,
    prefix: str = "\n    x=",
    suffix: str = ","
) -> str:
    """格式化特征向量为可读字符串"""
    return (
        prefix
        + np.array2string(
            x,
            max_line_width=80,      # 每行最大80字符
            threshold=10,            # 超过10个元素时省略中间部分
            prefix=prefix,           # 用于计算缩进
            suffix=suffix,
            precision=3,             # 保留3位小数
        )
        + suffix
    )
```

**输出示例**：

```python
# 短向量
x = np.array([0.1, 0.2, 0.3])
# → "\n    x=[0.1 0.2 0.3],"

# 长向量（超过threshold）
x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
# → "\n    x=[0.1 0.2 0.3 ... 1.  1.1 1.2],"
```

---

### 3.3 📐 Instance 基类详解 (Lines 38-163)

#### 3.3.1 构造函数 (Lines 47-69)

```python
def __init__(
    self,
    schema: "Schema",                                    # 流结构描述
    instance: Union[InstanceExample, FeatureVector]      # Java对象 或 NumPy数组
) -> None:
    self._schema: "Schema" = schema
    self._java_instance: Optional[InstanceExample] = None  # 惰性缓存
    self._x: Optional[FeatureVector] = None                # 惰性缓存

    # 根据输入类型初始化对应的缓存
    if isinstance(instance, InstanceExample):
        self._java_instance = instance   # 来自 MOA 流
    elif isinstance(instance, np.ndarray):
        self._x = instance               # 来自 Python 流
    else:
        raise ValueError(f"Given instance type unsupported: {type(instance)}")
```

**内部状态模型**：

```
初始化后的状态：

情况1：来自 Java
┌─────────────────────────────────┐
│ _schema = Schema(...)           │
│ _java_instance = <MOA对象>      │  ✅ 已填充
│ _x = None                       │  ❌ 空（惰性）
└─────────────────────────────────┘

情况2：来自 Python
┌─────────────────────────────────┐
│ _schema = Schema(...)           │
│ _java_instance = None           │  ❌ 空（惰性）
│ _x = array([0.1, 0.2, ...])    │  ✅ 已填充
└─────────────────────────────────┘
```

#### 3.3.2 特征向量属性 (Lines 113-125) ⭐ 核心

```python
@property
def x(self) -> FeatureVector:
    """返回特征向量，惰性求值"""

    # 优先级1：Python 原生数据
    if self._x is not None:
        return self._x

    # 优先级2：从 Java 实例提取
    elif self._java_instance is not None:
        moa_instance = self.java_instance.getData()  # 获取 MOA Instance
        self._x = np.empty(moa_instance.numInputAttributes())

        # 逐个提取特征值（Java → Python）
        for i in range(0, moa_instance.numInputAttributes()):
            self._x[i] = moa_instance.value(i)

        return self._x

    else:
        raise ValueError("Instance has no feature vector")
```

**惰性求值流程图**：

```
首次调用 instance.x:
┌─────────────────────────────────────────────────────┐
│ if _x is not None:         → 直接返回（O(1)）       │
│     return _x                                       │
│                                                     │
│ elif _java_instance:                                │
│     moa_inst = _java_instance.getData()            │
│     _x = np.empty(d)                               │
│     for i in range(d):      → 逐个提取（O(d)）     │
│         _x[i] = moa_inst.value(i)                  │
│     return _x               → 缓存后返回            │
└─────────────────────────────────────────────────────┘

后续调用 instance.x:
┌─────────────────────────────────────────────────────┐
│ if _x is not None:         → 直接返回缓存（O(1)）  │
│     return _x                                       │
└─────────────────────────────────────────────────────┘
```

#### 3.3.3 Java 实例属性 (Lines 135-155) ⭐ 核心

```python
@property
def java_instance(self) -> InstanceExample:
    """返回 MOA Java 对象，惰性求值"""

    # 优先级1：已有 Java 实例
    if self._java_instance is not None:
        return self._java_instance

    # 优先级2：从 Python 数据构建
    elif self._x is not None:
        assert self._x.ndim == 1, "Feature vector must be 1D."

        # 1️⃣ 分配 Java double 数组（特征数 + 1 用于目标变量）
        jx = JArray(JDouble)(len(self.x) + 1)

        # 2️⃣ 复制特征值到 Java 数组
        jx[: len(self.x)] = self.x

        # 3️⃣ 创建 MOA DenseInstance（权重=1.0）
        instance = DenseInstance(1.0, jx)

        # 4️⃣ 关联数据集头信息（Schema）
        instance.setDataset(self.schema.get_moa_header())

        # 5️⃣ 设置目标变量（由 _set_y 处理）
        self._java_instance = InstanceExample(self._set_y(instance))

        return self._java_instance
```

**Python → Java 转换详解**：

```
Python 状态:
┌─────────────────────────────────────────┐
│ _x = np.array([0.1, 0.2, 0.3])         │  3个特征
│ _schema = Schema(3 features, 2 classes) │
│ _y_index = 1 (如果是 LabeledInstance)   │
└─────────────────────────────────────────┘
                    │
                    ▼ JArray(JDouble)(4)
┌─────────────────────────────────────────┐
│ jx = [0.1, 0.2, 0.3, ?]                │  4个位置（特征+目标）
└─────────────────────────────────────────┘
                    │
                    ▼ jx[:3] = self.x
┌─────────────────────────────────────────┐
│ jx = [0.1, 0.2, 0.3, ?]                │  前3个位置填充
└─────────────────────────────────────────┘
                    │
                    ▼ DenseInstance(1.0, jx)
┌─────────────────────────────────────────┐
│ MOA DenseInstance                       │
│   weight = 1.0                          │
│   values = [0.1, 0.2, 0.3, ?]          │
└─────────────────────────────────────────┘
                    │
                    ▼ setDataset(moa_header)
┌─────────────────────────────────────────┐
│ MOA DenseInstance (已关联 Schema)        │
│   knows: 3 input attrs, 1 class attr   │
│   knows: class values = ["yes", "no"]  │
└─────────────────────────────────────────┘
                    │
                    ▼ _set_y(instance) [子类覆盖]
┌─────────────────────────────────────────┐
│ LabeledInstance._set_y:                 │
│   instance.setClassValue(1)            │
│   → values = [0.1, 0.2, 0.3, 1.0]     │
│                                         │
│ Instance._set_y (基类):                 │
│   instance.setMissing(classIndex)      │
│   → values = [0.1, 0.2, 0.3, NaN]     │
└─────────────────────────────────────────┘
                    │
                    ▼ InstanceExample(instance)
┌─────────────────────────────────────────┐
│ MOA InstanceExample (最终包装)           │
│   可传入 MOA 学习器使用                  │
└─────────────────────────────────────────┘
```

#### 3.3.4 模板方法 _set_y (Lines 127-133)

```python
def _set_y(self, instance: DenseInstance) -> DenseInstance:
    """基类默认行为：将目标变量设为缺失"""
    instance.setMissing(instance.classIndex())  # 标记为 missing
    return instance
```

> 🔑 **设计模式：模板方法（Template Method）**
>
> 基类定义算法骨架（`java_instance` 属性），但将目标变量的设置委托给子类覆盖的 `_set_y` 方法。

---

### 3.4 📐 LabeledInstance 详解 (Lines 166-270)

#### 3.4.1 构造函数 (Lines 189-197)

```python
def __init__(
    self,
    schema: "Schema",
    instance: Union[InstanceExample, Tuple[FeatureVector, LabelIndex]],  # 支持元组
) -> None:
    self._y_index: Optional[LabelIndex] = None

    # 如果输入是 (x, y_index) 元组，解包
    if isinstance(instance, tuple):
        instance, self._y_index = instance  # instance 变为 x，self._y_index 被赋值

    super().__init__(schema, instance)
```

**输入格式对比**：

| 输入类型 | 示例 | 来源 |
|----------|------|------|
| `InstanceExample` | `<MOA Java对象>` | MOA 流 (ARFFStream) |
| `Tuple[ndarray, int]` | `(np.array([0.1, 0.2]), 1)` | Python 流 (NumpyStream) |

#### 3.4.2 工厂方法 from_array (Lines 199-237)

```python
@classmethod
def from_array(
    cls,
    schema: "Schema",
    x: FeatureVector,
    y_index: LabelIndex
) -> "LabeledInstance":
    """从 NumPy 数组和类别索引创建实例"""
    return cls(schema, (x, int(y_index)))  # 封装为元组传入构造函数
```

> 💡 **为什么用工厂方法而不是直接构造？**
>
> - 提供更清晰的 API：`from_array(schema, x, y_index)` 比 `LabeledInstance(schema, (x, y_index))` 更易读
> - 可以在工厂方法中添加验证逻辑
> - 遵循 Python 惯例（如 `datetime.fromtimestamp()`）

#### 3.4.3 标签属性 (Lines 239-256)

```python
@property
def y_label(self) -> Label:
    """返回类别标签字符串"""
    return self.schema.get_value_for_index(self.y_index)
    # 例：y_index=0 → schema查表 → "yes"

@property
def y_index(self) -> LabelIndex:
    """返回类别索引（惰性求值）"""
    if self._y_index is not None:
        return self._y_index
    elif self._java_instance is not None:
        # 从 Java 实例提取
        self._y_index = int(self.java_instance.getData().classValue())
        return self._y_index
    else:
        raise ValueError(f"{self.__class__.__name__} must have a y_index.")
```

**y_index vs y_label 对比**：

| 属性 | 类型 | 示例 | 用途 |
|------|------|------|------|
| `y_index` | `int` | `0`, `1`, `2` | 算法内部计算、数组索引 |
| `y_label` | `str` | `"yes"`, `"no"`, `"spam"` | 人类可读、日志输出 |

#### 3.4.4 覆盖 _set_y (Lines 258-260)

```python
def _set_y(self, instance: DenseInstance) -> DenseInstance:
    """覆盖：设置分类标签"""
    instance.setClassValue(self.y_index)  # 设置为类别索引
    return instance
```

---

### 3.5 📐 RegressionInstance 详解 (Lines 273-365)

结构与 `LabeledInstance` 类似，主要区别：

| 对比项 | LabeledInstance | RegressionInstance |
|--------|-----------------|-------------------|
| 目标变量 | `y_index: int` (离散) | `y_value: float` (连续) |
| 额外属性 | `y_label: str` | 无 |
| _set_y | `setClassValue(y_index)` | `setClassValue(y_value)` |

```python
@property
def y_value(self) -> TargetValue:
    """返回回归目标值（惰性求值）"""
    if self._y_value is not None:
        return self._y_value
    elif self._java_instance is not None:
        self._y_value = self.java_instance.getData().classValue()  # 浮点数
        return self._y_value
    else:
        raise ValueError(f"{self.__class__.__name__} must have a y_value.")
```

---

## 4. 数据流分析

### 4.1 数据来源

```
┌───────────────────────────────────────────────────────────────────┐
│                       数据来源分类                                 │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1️⃣ MOA Java 流 (ARFFStream, MOAStream)                          │
│     ┌─────────────────────────────────────────────────────────┐  │
│     │ java_stream.nextInstance()                              │  │
│     │     ↓                                                   │  │
│     │ InstanceExample (Java 对象)                             │  │
│     │     ↓                                                   │  │
│     │ LabeledInstance(schema, java_instance)                  │  │
│     │     → _java_instance = java_instance                   │  │
│     │     → _x = None (惰性)                                  │  │
│     └─────────────────────────────────────────────────────────┘  │
│                                                                   │
│  2️⃣ Python 流 (NumpyStream, CSVStream)                           │
│     ┌─────────────────────────────────────────────────────────┐  │
│     │ python_stream.next_instance()                           │  │
│     │     ↓                                                   │  │
│     │ (x: ndarray, y_index: int)                             │  │
│     │     ↓                                                   │  │
│     │ LabeledInstance.from_array(schema, x, y_index)          │  │
│     │     → _x = x                                            │  │
│     │     → _y_index = y_index                                │  │
│     │     → _java_instance = None (惰性)                      │  │
│     └─────────────────────────────────────────────────────────┘  │
│                                                                   │
│  3️⃣ Stream Wrapper (OpenFeatureStream)                           │
│     ┌─────────────────────────────────────────────────────────┐  │
│     │ wrapper.next_instance()                                 │  │
│     │     ↓                                                   │  │
│     │ base_instance = base_stream.next_instance()            │  │
│     │ x_subset = base_instance.x[active_indices]             │  │
│     │     ↓                                                   │  │
│     │ LabeledInstance.from_array(schema, x_subset, y_index)   │  │
│     │ new_instance.feature_indices = active_indices          │  │
│     └─────────────────────────────────────────────────────────┘  │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### 4.2 数据消费

```
┌───────────────────────────────────────────────────────────────────┐
│                       数据消费者分类                               │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1️⃣ Python 学习器 (FOBOSClassifier, FTRLClassifier, ...)         │
│     ┌─────────────────────────────────────────────────────────┐  │
│     │ learner.train(instance)                                 │  │
│     │     ↓                                                   │  │
│     │ x = instance.x              # 访问特征向量              │  │
│     │ y = instance.y_index        # 访问标签                  │  │
│     │     ↓                                                   │  │
│     │ # 纯 NumPy 计算，无需 Java                              │  │
│     │ W[indices] -= eta * grad * values                      │  │
│     └─────────────────────────────────────────────────────────┘  │
│                                                                   │
│  2️⃣ MOA 学习器 (HoeffdingTree, NaiveBayes, ...)                  │
│     ┌─────────────────────────────────────────────────────────┐  │
│     │ learner.train(instance)                                 │  │
│     │     ↓                                                   │  │
│     │ java_inst = instance.java_instance  # 触发惰性转换      │  │
│     │     ↓                                                   │  │
│     │ moa_learner.trainOnInstance(java_inst)  # 传给 Java    │  │
│     └─────────────────────────────────────────────────────────┘  │
│                                                                   │
│  3️⃣ 评估器 (ClassificationEvaluator, ...)                        │
│     ┌─────────────────────────────────────────────────────────┐  │
│     │ evaluator.update(true_label, prediction)                │  │
│     │     ↓                                                   │  │
│     │ true_label = instance.y_index                          │  │
│     │ # 仅访问标签，不需要完整实例                            │  │
│     └─────────────────────────────────────────────────────────┘  │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### 4.3 惰性求值的性能优势

```
场景：Python 学习器处理来自 MOA 流的数据

无惰性求值（假设立即转换）:
┌─────────────────────────────────────────────────────────────┐
│ for instance in moa_stream:    # 每次迭代                    │
│     x = java_to_numpy(inst)    # O(d) 转换 ← 立即执行      │
│     java = numpy_to_java(x)    # O(d) 转换 ← 也立即执行    │
│     learner.train(instance)    # 只用 x，java 被浪费！      │
│                                                             │
│ 总计：2 * O(d) * n 次转换                                   │
└─────────────────────────────────────────────────────────────┘

有惰性求值（当前实现）:
┌─────────────────────────────────────────────────────────────┐
│ for instance in moa_stream:    # 每次迭代                    │
│     # _java_instance 已填充，_x = None                      │
│     learner.train(instance)                                 │
│         x = instance.x         # 首次访问时才转换 O(d)     │
│         # _java_instance 从未被访问，无需反向转换          │
│                                                             │
│ 总计：O(d) * n 次转换（减少 50%）                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 依赖关系

### 5.1 外部库依赖

| 库 | 导入语句 | 用途 |
|-----|----------|------|
| `numpy` | `import numpy as np` | 特征向量存储、数组操作 |
| `jpype` | `from jpype import JArray, JDouble` | Python-Java 数组桥接 |
| `typing` | `from typing import ...` | 类型注解 |

### 5.2 MOA Java 依赖（通过 JPype）

| Java 类 | 导入语句 | 用途 |
|---------|----------|------|
| `DenseInstance` | `from com.yahoo.labs.samoa.instances import DenseInstance` | MOA 密集数据实例 |
| `InstanceExample` | `from moa.core import InstanceExample` | MOA 实例包装器 |

### 5.3 内部模块依赖

```
instance.py
    │
    ├── openmoa.type_alias
    │   ├── FeatureVector  = NDArray[double]
    │   ├── Label          = str
    │   ├── LabelIndex     = int
    │   └── TargetValue    = double
    │
    └── openmoa.stream.Schema (仅类型注解，运行时不导入)
```

### 5.4 被依赖关系（30个文件引用）

```
instance.py
    │
    ├── openmoa.stream.*          (流模块)
    │   ├── _stream.py            (创建实例)
    │   ├── stream_wrapper.py     (包装实例)
    │   └── torch.py              (PyTorch 流)
    │
    ├── openmoa.classifier.*      (分类器)
    │   ├── _fobos_classifier.py
    │   ├── _ftrl_classifier.py
    │   └── ... (12个分类器)
    │
    ├── openmoa.regressor.*       (回归器)
    │   ├── _fesl_regressor.py
    │   └── _oasf_regressor.py
    │
    ├── openmoa.base.*            (基类)
    │   ├── _classifier.py
    │   └── _regressor.py
    │
    └── openmoa.evaluation.*      (评估)
        └── evaluation.py
```

---

## 6. 特殊机制与技术点

### 6.1 🎨 设计模式

#### 6.1.1 惰性求值模式（Lazy Evaluation）

```python
@property
def x(self) -> FeatureVector:
    if self._x is not None:      # 缓存命中
        return self._x
    elif self._java_instance is not None:
        self._x = self._extract_from_java()  # 首次访问时计算
        return self._x
```

**优点**：
- 避免不必要的计算（如果从不访问 `.x`，就不执行转换）
- 透明缓存（调用者无需关心是否已转换）

#### 6.1.2 模板方法模式（Template Method）

```python
# 基类定义算法骨架
class Instance:
    @property
    def java_instance(self):
        # ...
        self._java_instance = InstanceExample(self._set_y(instance))  # 钩子
        # ...

    def _set_y(self, instance):  # 默认实现
        instance.setMissing(instance.classIndex())
        return instance

# 子类覆盖钩子方法
class LabeledInstance(Instance):
    def _set_y(self, instance):  # 覆盖
        instance.setClassValue(self.y_index)
        return instance
```

#### 6.1.3 工厂方法模式（Factory Method）

```python
@classmethod
def from_array(cls, schema, x, y_index):
    """静态工厂方法"""
    return cls(schema, (x, int(y_index)))

@classmethod
def from_java_instance(cls, schema, java_instance):
    """另一个工厂方法"""
    return cls(schema, java_instance)
```

### 6.2 🔐 类型安全机制

#### 6.2.1 类型别名（Type Alias）

```python
# type_alias.py
FeatureVector = NDArray[double]  # 比 np.ndarray 更具体
LabelIndex = int                 # 语义化命名
```

#### 6.2.2 条件导入避免循环依赖

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 仅类型检查时为 True
    from openmoa.stream import Schema  # 运行时不执行

def __init__(self, schema: "Schema", ...):  # 字符串形式的前向引用
    ...
```

### 6.3 ⚡ 性能优化技巧

#### 6.3.1 缓存机制

```python
# 属性只计算一次，后续访问直接返回缓存
self._x: Optional[FeatureVector] = None

@property
def x(self):
    if self._x is not None:
        return self._x  # O(1) 缓存命中
    # ... 计算 ...
    self._x = result    # 存入缓存
    return self._x
```

#### 6.3.2 批量数组操作

```python
# Java → Python 转换
jx[: len(self.x)] = self.x  # NumPy 批量赋值，比 for 循环快

# 但提取时仍需循环（受 JPype 限制）
for i in range(0, moa_instance.numInputAttributes()):
    self._x[i] = moa_instance.value(i)
```

### 6.4 🛡️ 错误处理

```python
# 类型检查
if isinstance(instance, InstanceExample):
    ...
elif isinstance(instance, np.ndarray):
    ...
else:
    raise ValueError(f"Given instance type unsupported: {type(instance)}")

# 断言
assert self._x.ndim == 1, "Feature vector must be 1D."

# 状态检查
if self._y_index is not None:
    return self._y_index
elif self._java_instance is not None:
    ...
else:
    raise ValueError(f"{self.__class__.__name__} must have a y_index.")
```

---

## 7. 潜在问题与改进建议

### 7.1 🐛 已识别问题

#### 问题 1：Java 转换的 for 循环瓶颈

```python
# 当前实现：逐元素提取
for i in range(0, moa_instance.numInputAttributes()):
    self._x[i] = moa_instance.value(i)
```

**风险**：对于高维数据（如文本特征 d=10000+），循环开销显著。

**建议改进**：
```python
# 探索 JPype 是否支持批量转换
# 或使用 MOA 的 toDoubleArray() 方法（如果存在）
values = moa_instance.toDoubleArray()  # 假设存在
self._x = np.array(values[:num_input_attrs])
```

#### 问题 2：动态属性的类型检查缺失

```python
# stream_wrapper.py 中动态添加属性
new_instance.feature_indices = valid_indices  # 没有类型声明

# 下游使用
if hasattr(instance, "feature_indices"):  # 运行时检查
    ...
```

**建议改进**：
```python
# 方案1：定义协议（Protocol）
from typing import Protocol

class SparseAwareInstance(Protocol):
    feature_indices: np.ndarray

# 方案2：使用 dataclass 或 attrs
@dataclass
class InstanceWithIndices:
    base: Instance
    feature_indices: np.ndarray
```

#### 问题 3：_set_y 方法命名不够清晰

```python
def _set_y(self, instance: DenseInstance) -> DenseInstance:
    """名称暗示是 setter，但实际是在构建 Java 实例时的钩子"""
```

**建议改进**：
```python
def _configure_target_attribute(self, instance: DenseInstance) -> DenseInstance:
    """更清晰地表达意图"""
```

### 7.2 📈 性能优化建议

#### 建议 1：预分配优化

```python
# 当前
self._x = np.empty(moa_instance.numInputAttributes())
for i in range(0, moa_instance.numInputAttributes()):
    self._x[i] = moa_instance.value(i)

# 改进：使用列表推导式 + 一次性转换
self._x = np.array([moa_instance.value(i) for i in range(n)], dtype=np.float64)
```

#### 建议 2：__slots__ 减少内存

```python
class Instance:
    __slots__ = ('_schema', '_java_instance', '_x')  # 约 40% 内存节省

    def __init__(self, ...):
        ...
```

### 7.3 🔍 可读性改进

#### 建议 1：提取常量

```python
# 当前
jx = JArray(JDouble)(len(self.x) + 1)  # +1 是什么？

# 改进
NUM_TARGET_ATTRIBUTES = 1  # MOA 实例的目标属性数
jx = JArray(JDouble)(len(self.x) + NUM_TARGET_ATTRIBUTES)
```

#### 建议 2：统一属性访问风格

```python
# 当前混用
self._y_index: Optional[LabelIndex] = None  # 下划线前缀
self._y_value: Optional[TargetValue] = None

# 建议统一使用 @cached_property (Python 3.8+)
from functools import cached_property

@cached_property
def y_index(self) -> LabelIndex:
    if self._java_instance:
        return int(self.java_instance.getData().classValue())
    raise ValueError(...)
```

---

## 8. 使用示例

### 8.1 🚀 基本用法

```python
from openmoa.datasets import ElectricityTiny
from openmoa.instance import LabeledInstance

# 从数据集获取实例
stream = ElectricityTiny()
instance: LabeledInstance = stream.next_instance()

# 访问特征向量
print(f"Features: {instance.x}")
# → Features: [0.    0.056 0.439 0.003 0.423 0.415]

# 访问类别信息
print(f"Class index: {instance.y_index}")  # → 1
print(f"Class label: {instance.y_label}")  # → "1"

# 访问 Schema
print(f"Num features: {instance.schema.get_num_attributes()}")  # → 6
print(f"Num classes: {instance.schema.get_num_classes()}")      # → 2
```

### 8.2 📊 手动创建实例

```python
from openmoa.stream import Schema
from openmoa.instance import LabeledInstance, RegressionInstance
import numpy as np

# 1. 创建 Schema
schema = Schema.from_custom(
    feature_names=["f1", "f2", "f3"],
    dataset_name="MyDataset",
    values_for_class_label=["positive", "negative"]
)

# 2. 创建分类实例
x = np.array([0.1, 0.2, 0.3])
labeled_inst = LabeledInstance.from_array(schema, x, y_index=0)

print(labeled_inst)
# → LabeledInstance(
# →     Schema(MyDataset),
# →     x=[0.1 0.2 0.3],
# →     y_index=0,
# →     y_label='positive'
# → )

# 3. 创建回归实例（需要不同的 Schema）
reg_schema = Schema.from_custom(
    feature_names=["f1", "f2"],
    dataset_name="RegressionData",
    target_type='numeric'
)
reg_inst = RegressionInstance.from_array(reg_schema, np.array([1.0, 2.0]), y_value=3.14)

print(f"Target value: {reg_inst.y_value}")  # → 3.14
```

### 8.3 🔄 与 MOA 学习器交互

```python
from openmoa.classifier import HoeffdingTree
from openmoa.datasets import ElectricityTiny

stream = ElectricityTiny()
schema = stream.get_schema()
learner = HoeffdingTree(schema=schema)

for i, instance in enumerate(stream):
    if i >= 100:
        break

    # 预测（内部访问 instance.java_instance）
    pred = learner.predict(instance)

    # 训练（内部访问 instance.java_instance）
    learner.train(instance)

    # instance.java_instance 在首次访问时惰性创建
    # 后续访问直接返回缓存
```

### 8.4 🔧 与 Python 学习器交互

```python
from openmoa.classifier._fobos_classifier import FOBOSClassifier
from openmoa.datasets import Magic04
from openmoa.stream import ShuffledStream, OpenFeatureStream

# 创建特征演化流
base = ShuffledStream(Magic04(), random_seed=42)
stream = OpenFeatureStream(base, evolution_pattern="tds", total_instances=1000)

schema = stream.get_schema()
learner = FOBOSClassifier(schema=schema, alpha=1.0)

for instance in stream:
    # Python 学习器只访问 .x 和 .y_index
    # 不触发 Java 转换

    x = instance.x                    # NumPy 数组
    y = instance.y_index              # 整数

    # 如果有 feature_indices，用于稀疏更新
    if hasattr(instance, 'feature_indices'):
        indices = instance.feature_indices
        # learner 内部使用 indices 进行稀疏权重更新

    pred = learner.predict(instance)
    learner.train(instance)
```

### 8.5 📋 调试与检查

```python
instance = stream.next_instance()

# 查看实例的完整表示
print(repr(instance))
# → LabeledInstance(
# →     Schema(Magic04),
# →     x=[0.123 0.456 ... 0.789],
# →     y_index=1,
# →     y_label='g'
# → )

# 检查内部状态
print(f"Has cached x: {instance._x is not None}")
print(f"Has cached java: {instance._java_instance is not None}")

# 强制触发转换
_ = instance.java_instance  # 触发 Python → Java
print(f"Java string: {instance.java_instance.toString()}")
# → "0.123,0.456,...,0.789,g,"
```

---

## 📎 附录

### A. 术语对照表

| 中文术语 | 英文术语 | 说明 |
|----------|----------|------|
| 实例 | Instance | 单个数据点 |
| 特征向量 | Feature Vector | 输入属性数组 |
| 类别索引 | Label Index / Class Index | 分类标签的数字表示 |
| 类别标签 | Label / Class Label | 分类标签的字符串表示 |
| 目标值 | Target Value | 回归任务的连续目标变量 |
| 惰性求值 | Lazy Evaluation | 延迟计算直到首次访问 |
| 模板方法 | Template Method | 定义算法骨架，子类覆盖具体步骤 |
| 工厂方法 | Factory Method | 创建对象的静态方法 |
| 循环导入 | Circular Import | 模块 A 导入 B，B 又导入 A |
| 前向引用 | Forward Reference | 用字符串引用尚未定义的类型 |

### B. 类型别名定义

```python
# openmoa/type_alias.py

FeatureVector = NDArray[double]
"""特征向量：双精度浮点数的一维 NumPy 数组"""

LabelIndex = int
"""类别索引：非负整数，表示标签在标签列表中的位置"""

LabelProbabilities = NDArray[double]
"""预测概率：双精度浮点数数组，各类别的预测概率"""

Label = str
"""类别标签：字符串形式的分类标签"""

TargetValue = double
"""目标值：回归任务中的连续因变量"""
```

### C. 相关文件快速索引

| 文件 | 路径 | 说明 |
|------|------|------|
| Instance 模块 | `src/openmoa/instance.py` | 本文件 |
| 类型别名 | `src/openmoa/type_alias.py` | 类型定义 |
| Stream 基类 | `src/openmoa/stream/_stream.py` | 创建 Instance 的源头 |
| Schema 类 | `src/openmoa/stream/_stream.py:49` | 流结构描述 |
| FOBOS 分类器 | `src/openmoa/classifier/_fobos_classifier.py` | 使用 Instance 的示例 |
| Stream Wrapper | `src/openmoa/stream/stream_wrapper.py` | 扩展 Instance 的示例 |

---

> 📝 **文档版本**: 1.0
> **最后更新**: 2026-02-11
> **作者**: Claude Code Analysis
