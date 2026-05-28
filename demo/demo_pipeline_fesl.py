# SPDX-License-Identifier: BSD-3-Clause
import sys
import os

# ================================================================
# 1. 核心修复：自动将 ../src 添加到 Python 搜索路径
# ================================================================
# 获取当前脚本所在目录 (例如: .../OpenMOA/demo)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (例如: .../OpenMOA)
project_root = os.path.dirname(current_dir)
# 拼接 src 路径 (例如: .../OpenMOA/src)
src_path = os.path.join(project_root, 'src')

if src_path not in sys.path:
    sys.path.insert(0, src_path)
    print(f"✅ 已添加源码路径: {src_path}")

# ================================================================
# 2. 引用部分 (根据你的描述修改了 Pipeline 的位置)
# ================================================================

# [修改点]：根据你的指示，pipeline 在 openmoa/stream/preprocessing 下
try:
    from openmoa.stream.preprocessing.pipeline import ClassifierPipeline
except ImportError:
    # 如果文件名不是 pipeline.py 而是直接在 __init__.py 里，尝试这个：
    from openmoa.stream.preprocessing import ClassifierPipeline

# 其他引用保持原样（假设它们位置没变）
from openmoa.classifier import FESLClassifier 
from openmoa.datasets import ElectricityTiny
from openmoa.evaluation import prequential_evaluation
from openmoa.drift.detectors import ADWIN

# ================================================================
# 3. 业务逻辑
# ================================================================

# 1. 准备数据流
stream = ElectricityTiny()

# 2. 准备算法
# 确保 FESLClassifier 也能被正确引用
fesl_learner = FESLClassifier(
    schema=stream.get_schema(),
    alpha=0.1,
    window_size=100
)

# 3. 准备漂移检测的回调函数
def label_not_equals_prediction(instance, prediction):
    label = instance.y_index
    return int(label != prediction)

# 4. 构建 Pipeline
print("正在构建 Pipeline...")
pipe = (
    ClassifierPipeline()
    # .add_transformer(normalisation_transformer) 
    .add_classifier(fesl_learner)
    .add_drift_detector(
        ADWIN(), 
        get_drift_detector_input_func=label_not_equals_prediction
    )
)

print(f"Pipeline Structure: {pipe}")

# ... 前面的代码不变 ...

# 5. 运行评估
print("开始运行 Prequential Evaluation...")
results = prequential_evaluation(
    stream=stream,
    learner=pipe, 
    max_instances=1000,
    window_size=100
)

print("\n--- 📊 评估结果 ---")

# 1. 累积准确率 (这个之前运行成功了)
if hasattr(results, 'cumulative') and results.cumulative is not None:
    print(f"Cumulative Accuracy (累积): {results.cumulative.accuracy():.4f}")

# 2. 滑动窗口准确率 [修改点]
if hasattr(results, 'windowed') and results.windowed is not None:
    # 获取准确率数据
    win_acc = results.windowed.accuracy()
    
    # 判断它是否是列表
    if isinstance(win_acc, list):
        # 如果是列表，取最后一个值 (最新的准确率)
        print(f"Windowed Accuracy (最近窗口): {win_acc[-1]:.4f}")
    else:
        # 如果不是列表，直接打印
        print(f"Windowed Accuracy (最近窗口): {win_acc:.4f}")
        
else:
    print("Windowed metric not found.")