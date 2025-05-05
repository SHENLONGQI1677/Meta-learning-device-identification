import numpy as np
import torch
from PIL.ImageOps import scale
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy
from IQ_imbalance_generate import generate_iq_inbalance_data
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(f"Using device: {device}")

innerstepsize = 0.0005

def gen_task(num_devices=8, train_samples=500):
    """Generate a meta-learning task with IQ imbalance data."""
    (train_data, train_labels) = generate_iq_inbalance_data(num_devices, train_samples)
    return train_data, train_labels  # 只返回训练数据

## 定义四分类模型
model = nn.Sequential(
    nn.Linear(8, 64),
    nn.ReLU(),  # 隐藏层仍然用 Tanh
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 8),  # 输出 4 个类别
    # 归一化为概率分布
).to(device)

# 交叉熵损失（适用于分类任务）
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=innerstepsize)


# 训练函数（修改为交叉熵损失）
def train_on_batch(x, y):
    x = totorch(x)
    y = torch.LongTensor(y).to(device)  # 目标标签转换为 LongTensor
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)  # 交叉熵损失
    loss.backward()
    optimizer.step()
    return loss.item()

# 预测函数
def predict(x):
    x = totorch(x)
    y_pred = model(x)
    return torch.argmax(y_pred, dim=1).cpu().numpy()  # 取概率最高的类别

def reset_parameters():
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def totorch(x):
    return ag.Variable(torch.Tensor(x).to(device), requires_grad=True) # **把数据放到 GPU**

global_accuracies = []

def calculate_accuracy(predicted, true):
    predict_flat = predicted.flatten()
    correct_count = np.sum(predict_flat == true)

    # 计算准确率
    accuracy = correct_count / len(true)

    return accuracy

import pandas as pd
data_records = []

accuracy_dict = {}
accuracy_dict_500 = {sample_num: [] for sample_num in range(10)}

for i in range(100):
    reset_parameters()
    for sample_num in range(10):
        x_test, y_test = gen_task(train_samples=500)
        test_scale = 1 - (sample_num + 1) / 500
        x_new_train, x_new_test, y_new_train, y_new_test = train_test_split(
            x_test, y_test, test_size=test_scale)

        # 初始化该sample_num的准确率列表
        if sample_num not in accuracy_dict:
            accuracy_dict[sample_num] = {}

        # 内循环训练
        for outerstep in range(1000):
            loss = train_on_batch(x_new_train, y_new_train)

            predicted = predict(x_new_test)
            predicted = abs(predicted.round())
            accuracy = calculate_accuracy(predicted, y_new_test)

            if outerstep == 999:  # 只记录第500次的准确率
                accuracy_at_500 = accuracy

        if accuracy_at_500 is not None:
            accuracy_dict_500[sample_num].append(accuracy_at_500)
            print(f"i:{i + 1}, sample_num: {sample_num}, 第500次 Accuracy: {accuracy_at_500:.3f}")

            # 收集数据
            if outerstep not in accuracy_dict[sample_num]:
                accuracy_dict[sample_num][outerstep] = []
            accuracy_dict[sample_num][outerstep].append(accuracy)


        print(f"i:{i + 1}, sample_nums: {sample_num} 最后一步Accuracy: {accuracy:.3f}")

# 平均化结果，并整理为DataFrame
avg_records = []

for sample_num, outer_dict in accuracy_dict.items():
    for outerstep, acc_list in outer_dict.items():
        avg_accuracy = np.mean(acc_list)
        avg_records.append({
            "sample_num": sample_num,
            "outerstep": outerstep + 1,  # 加1是为了从1开始
            "average_accuracy": avg_accuracy
        })

# 保存为 CSV 文件
avg_df = pd.DataFrame(avg_records)
avg_df.to_csv("accuracy_log.csv", index=False)

print("平均准确率数据已保存到 accuracy_log.csv")

final_records = []
for sample_num, acc_list in accuracy_dict_500.items():
    avg_accuracy = np.mean(acc_list)
    final_records.append({
        "sample_num": sample_num,
        "average_accuracy_at_step_500": avg_accuracy
    })

df = pd.DataFrame(final_records)
df.to_csv("accuracy_500_log.csv", index=False)

print("第500次训练的平均准确率已保存到 accuracy_500_log.csv")
# with open("my_data_folder/Iteration 400", "r", encoding="utf-8") as file:
#     Iteration_400 = json.load(file)


# plt.plot(Iteration_400, label='Iteration 400 in meta_learning', color='orange')
# plt.plot(all_accuracy, label='Without meta_learning', color='green')
# plt.xlim(0, 20)
# plt.xticks(range(20))
# plt.title('Comparison in same Epochs ')
# plt.xlabel('Epochs')



# plt.ylim(0, 1)
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
# plt.savefig("comparison in same Epochs.png", dpi=300)
# plt.show()