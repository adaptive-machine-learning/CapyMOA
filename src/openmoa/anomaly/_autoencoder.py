# SPDX-License-Identifier: BSD-3-Clause
from openmoa.base import AnomalyDetector
from openmoa.instance import Instance
import torch
import torch.nn as nn
import torch.optim as optim


class Autoencoder(AnomalyDetector):  # 定义一个Autoencoder类，继承自AnomalyDetector
    """Autoencoder anomaly detector

    This is a simple autoencoder anomaly detector that uses a single hidden layer.

    Reference:

    `Contextual One-Class Classification in Data Streams.
    Richard Hugh Moulton, Herna L. Viktor, Nathalie Japkowicz, and João Gama.
    arXiv:1907.04233, 2019.
    <https://arxiv.org/pdf/1907.04233>`_

    Example:
    >>> from openmoa.datasets import ElectricityTiny
    >>> from openmoa.anomaly import Autoencoder
    >>> from openmoa.evaluation import AnomalyDetectionEvaluator
    >>> stream = ElectricityTiny()  # 创建一个ElectricityTiny对象，对应数据集的数据流对象
    >>> schema = stream.get_schema()  # 获取数据类型和形状的描述
    >>> learner = Autoencoder(schema=schema)   # 创建一个Autoencoder对象
    >>> evaluator = AnomalyDetectionEvaluator(schema)  # 创建一个AnomalyDetectionEvaluator对象
    >>> while stream.has_more_instances(): # 循环遍历数据集
    ...     instance = stream.next_instance() # 获取下一个实例
    ...     proba = learner.score_instance(instance)  # 计算实例的异常分数
    ...     evaluator.update(instance.y_index, proba)  # 更新评估器
    ...     learner.train(instance)  # 训练模型
    >>> auc = evaluator.auc()  # 计算AUC分数
    >>> print(f"AUC: {auc:.2f}")
    AUC: 0.58

    """

    def __init__(  # 初始化方法，接受一个Schema对象和隐藏层大小，默认隐藏层大小为 2
        self,
        schema=None, # 数据类型和形状的描述
        hidden_layer=2, # 隐藏层大小
        learning_rate=0.5, # 学习率
        threshold=0.6, # 异常阈值
        random_seed=1, # 随机种子
    ):
        """Construct an Autoencoder anomaly detector # 构造一个Autoencoder异常检测器

        Parameters
        :param schema: The schema of the input data # 数据类型和形状的描述
        :param hidden_layer: Number of neurons in the hidden layer. The number should less than the number of input
        features. # 隐藏层大小
        :param learning_rate: Learning rate # 学习率
        :param threshold: Anomaly threshold # 异常阈值
        :param random_seed: Random seed # 随机种子
        """

        super().__init__(schema, random_seed=random_seed)  # 调用父类AnomalyDetector的初始化方法
        self.hidden_layer = hidden_layer # 隐藏层大小
        self.learning_rate = learning_rate # 学习率
        self.threshold = threshold # 异常阈值

        if self.hidden_layer >= self.schema.get_num_attributes(): # 如果隐藏层大小大于输入特征数量，抛出异常
            raise ValueError(
                "The number of hidden layer should be less than the number of input features"
            )
        torch.manual_seed(self.random_seed) # 设置随机种子
        self._initialise() # 初始化模型

    def _initialise(self): # 初始化模型
        class _AEModel(nn.Module): # 定义一个_AEModel类，继承自nn.Module
            def __init__(self, input_size, hidden_size):
                super(_AEModel, self).__init__() # 调用父类nn.Module的初始化方法
                self.encoder = nn.Sequential( # 定义一个编码器，输入特征数量为input_size，输出特征数量为hidden_size
                    nn.Linear(input_size, hidden_size, dtype=torch.double), nn.Sigmoid() # 定义一个线性层，输入特征数量为input_size，输出特征数量为hidden_size，使用双精度浮点数，使用Sigmoid激活函数
                )
                self.decoder = nn.Sequential( # 定义一个解码器，输入特征数量为hidden_size，输出特征数量为input_size
                    nn.Linear(hidden_size, input_size, dtype=torch.double), nn.Sigmoid() # 定义一个线性层，输入特征数量为hidden_size，输出特征数量为input_size，使用双精度浮点数，使用Sigmoid激活函数
                )

            def forward(self, x): # 定义前向传播方法，接受一个输入张量x，返回一个输出张量
                x = self.encoder(x) # 将输入张量x通过编码器
                x = self.decoder(x) # 将输出张量x通过解码器
                return x

        self.model = _AEModel( # 创建一个_AEModel对象
            input_size=self.schema.get_num_attributes(), hidden_size=self.hidden_layer # 输入特征数量为self.schema.get_num_attributes()，隐藏层大小为self.hidden_layer
        )
        self.criterion = nn.MSELoss() # 定义一个均方误差损失函数
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate) # 定义一个随机梯度下降优化器

    def __str__(self):  # 定义一个字符串方法，返回一个字符串
        return "Autoencoder Anomaly Detector" # 返回一个字符串

    def train(self, instance: Instance): # 定义一个训练方法，接受一个实例对象
        # Convert the input to a tensor
        input = torch.from_numpy(instance.x) # 将输入张量x转换为张量

        # Forward pass
        self.optimizer.zero_grad() # 将优化器的梯度清零
        output = self.model(input) # 将输入张量x通过模型

        # Compute the loss
        loss = self.criterion(output, input) # 计算损失 ，输出张量output和输入张量input

        # Backward pass and optimization
        loss.backward() # 计算梯度
        self.optimizer.step() # 更新模型参数

    def predict(self, instance: Instance) -> int: # 定义一个预测方法，接受一个实例对象，返回一个整数
        if self.score_instance(instance) > 0.5: # 如果评分大于0.5，返回0
            return 0
        else:
            return 1 # 否则返回1

    def score_instance(self, instance: Instance) -> float: # 定义一个评分方法，接受一个实例对象，返回一个浮点数
        # Convert the input to a tensor
        input = torch.from_numpy(instance.x) # 将输入张量x转换为张量

        # Pass the input through the autoencoder
        output = self.model(input) # 将输入张量x通过模型

        # Compute the reconstruction error
        error = torch.mean(torch.square(input - output)) # 计算重建误差 ，输入张量input和输出张量output

        return 2.0 ** (-(error.item() / self.threshold)) # 返回一个浮点数
