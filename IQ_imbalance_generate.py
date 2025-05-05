# 导入必要的库
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt

# 测试 Meta-Learning 任务生成
(train_data, train_labels) = generate_iq_inbalance_data(train_samples=10)

# print("train_data:\n", train_data[:100])  # 只打印前 10 个，避免过长
# print("train_labels:\n", train_labels[:100])
