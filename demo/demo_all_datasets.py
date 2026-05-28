# SPDX-License-Identifier: BSD-3-Clause
import sys
import os
import time

# 强制将 src 目录加入路径，确保加载本地代码
sys.path.insert(0, os.path.abspath('./src'))

try:
    # 尝试导入本地包
    import openmoa.datasets as datasets
    from openmoa.stream import MOAStream
    
    print(f"✅ Successfully loaded local package from: {os.path.dirname(datasets.__file__)}")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("请检查项目根目录下是否存在 'src/openmoa' 文件夹。")
    sys.exit(1)

def check_list(title, dataset_list, is_binary_group=True):
    """
    辅助函数：测试一组数据集
    """
    print(f"\n🚀 Starting Check for {title} ({len(dataset_list)} datasets)...")
    print(f"{'Dataset Name':<15} | {'Status':<10} | {'Samples':<10} | {'Features':<10} | {'Classes':<10}")
    print("-" * 70)

    failures = []

    for dataset_cls in dataset_list:
        name = dataset_cls.__name__
        try:
            # 1. 初始化流 (触发下载/读取)
            stream: MOAStream = dataset_cls()
            
            # 2. 读取一条数据 (验证解析)
            instance = stream.next_instance()
            
            # 获取统计信息
            n_samples = len(stream) if hasattr(stream, '__len__') else "Unknown"
            n_features = stream.schema.get_num_attributes()
            n_classes = stream.schema.get_num_classes()
            
            status_icon = "✅ PASS"
            
            # 简单的类别数量检查
            warning_msg = ""
            if is_binary_group and n_classes != 2:
                status_icon = "⚠️ WARN"
                warning_msg = f" (Expected 2, got {n_classes})"
            elif not is_binary_group and n_classes < 3:
                # 多分类组里如果只有2类，提示一下（比如 InternetAds 清洗后是2类）
                status_icon = "⚠️ NOTE" 
                warning_msg = f" (Binary?)"

            # 打印行
            print(f"{name:<15} | {status_icon:<10} | {str(n_samples):<10} | {str(n_features):<10} | {str(n_classes):<10}{warning_msg}")

        except Exception as e:
            print(f"{name:<15} | ❌ FAIL    | {'-':<10} | {'-':<10} | {'-':<10}")
            print(f"  └── Error: {e}")
            failures.append(name)
            
    print("-" * 70)
    return failures

def test_all_datasets():
    # ==========================================
    # 1. Binary Classification Benchmarks (10)
    # ==========================================
    binary_datasets = [
        datasets.RCV1,
        datasets.W8a,
        datasets.Adult,
        datasets.InternetAds, 
        datasets.Magic04,
        datasets.Spambase,
        datasets.Musk,
        datasets.SVMGuide3,
        datasets.German,
        datasets.Australian,
        datasets.Ionosphere
    ]

    # ==========================================
    # 2. Multi-Class Classification Benchmarks (7)
    # ==========================================
    multiclass_datasets = [
        datasets.Covtype,   # 7 classes
        datasets.DryBean,     # 7 classes
        datasets.Optdigits,   # 10 classes
        datasets.Frogs,       # 4 classes (Family)
        datasets.Wine,        # 3 classes
        datasets.Splice       # 3 classes (Moved from Binary)
    ]

    # 执行测试
    failed_bin = check_list("Binary Datasets", binary_datasets, is_binary_group=True)
    failed_multi = check_list("Multi-Class Datasets", multiclass_datasets, is_binary_group=False)

    # 总结
    total_failed = len(failed_bin) + len(failed_multi)
    if total_failed == 0:
        print("\n🎉 CONGRATULATIONS! All 17 datasets are ready for experiments.")
    else:
        print(f"\n⚠️ Summary: {total_failed} datasets failed to load.")
        if failed_bin: print(f"  - Binary Failures: {failed_bin}")
        if failed_multi: print(f"  - Multi-Class Failures: {failed_multi}")

if __name__ == "__main__":
    test_all_datasets()