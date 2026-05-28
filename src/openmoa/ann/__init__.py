# SPDX-License-Identifier: BSD-3-Clause
"""Artificial Neural Networks for OpenMOA."""

from ._perceptron import Perceptron  # 加载Perceptron类

__all__ = [     # 导出Perceptron类，以便在其他地方使用
# __all__是Python模块的特殊变量，用于导出模块中的所有公共（公开）变量
# 当使用from openmoa.ann import *时，Python只会导入__all__列表中指定的名称
    "Perceptron",
]
