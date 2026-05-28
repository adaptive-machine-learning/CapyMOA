# SPDX-License-Identifier: BSD-3-Clause
from openmoa.stream._stream import Schema
from torch import nn
from torch import Tensor


class Perceptron(nn.Module):  # 定义一个Perceptron类，继承自nn.Module
    """A simple feedforward neural network with one hidden layer."""
    # 初始化方法，接受一个Schema对象和隐藏层大小，默认隐藏层大小为50
    def __init__(self, schema: Schema, hidden_size: int = 50): # schema是数据类型和形状的描述
        """Initialize the model.

        :param schema: Schema describing the data types and shapes.
        :param hidden_size: Number of hidden units in the first layer.
        """
        super(Perceptron, self).__init__() # 调用父类nn.Module的初始化方法
        in_features = schema.get_num_attributes()  # 获取输入特征数量
        out_features = schema.get_num_classes()    # 获取输出特征数量
        self._fc1 = nn.Linear(in_features, hidden_size)  # 定义第一个全连接层，输入特征数量为in_features，输出特征数量为hidden_size
        self._relu = nn.ReLU()  # 定义一个ReLU激活函数
        self._fc2 = nn.Linear(hidden_size, out_features)  # 定义第二个全连接层，输入特征数量为hidden_size，输出特征数量为out_features

    def forward(self, x: Tensor) -> Tensor:  # 定义前向传播方法，接受一个输入张量x，返回一个输出张量
        """Forward pass through the network.

        :param x: Input tensor of shape ``(batch_size, num_features)``.
        :return: Output tensor of shape ``(batch_size, num_classes)``.
        """
        x = self._fc1(x) # 将输入张量x通过第一个全连接层
        x = self._relu(x) # 将输出张量x通过ReLU激活函数
        x = self._fc2(x) # 将输出张量x通过第二个全连接层
        return x
