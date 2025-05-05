import numpy as np
import torch
from sympy.diffgeom import Differential
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy
from IQ_imbalance_generate import generate_iq_inbalance_data
from sklearn.model_selection import train_test_split
import json
import os
import csv

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(f"Using device: {device}")

# Meta-learning parameters

innerstepsize = 0.005
innerepochs = 50
outerstepsize0 = 0.01
niterations = 800

# seed = 0
# torch.manual_seed(seed)
# np.random.seed(seed)

folder_path = "./meta_learning_results"  # 存储路径
os.makedirs(folder_path, exist_ok=True)  # 创建主目录

num_devices = 8

# 生成任务
def gen_task(num_devices=num_devices, train_samples=500):
    """Generate a meta-learning task with IQ imbalance data."""
    (train_data, train_labels) = generate_iq_inbalance_data(num_devices, train_samples)
    return train_data, train_labels  # 只返回训练数据


## 定义四分类模型
model = nn.Sequential(
    nn.Linear(8, 64),
    nn.ReLU(), # 隐藏层仍然用 Tanh
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, num_devices),  # 输出 decives 个类别
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



def totorch(x):
    return ag.Variable(torch.Tensor(x).to(device), requires_grad=True) # **把数据放到 GPU**



def calculate_accuracy(predicted, true):
    predict_flat = predicted.flatten()
    correct_count = np.sum(predict_flat == true)

    # 计算准确率
    accuracy = correct_count / len(true)

    return accuracy


global_accuracies = []
Different_sample_nums_accuracies={}

# Reptile 训练循环
for iteration in range(niterations):
    weights_before = deepcopy(model.state_dict())

    # 生成任务
    x_all, y_all = gen_task()

    # **确保 batch_size 至少为 1**
    batch_size = max(1, len(x_all) // 10)
    inds = np.random.permutation(len(x_all))

    # 训练
    for _ in range(innerepochs):
        for start in range(0, len(x_all), batch_size):
            mbinds = inds[start:start + batch_size]
            loss = train_on_batch(x_all[mbinds], y_all[mbinds])

    # **Meta-update**
    weights_after = model.state_dict()
    outerstepsize = outerstepsize0 * (1 - iteration / niterations)
    model.load_state_dict(
        {name: weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize for name in
         weights_before})


    # **Meta-检验并计算准确率**
    if (iteration + 1) % 800 == 0:
        iteration_folder = os.path.join(folder_path, f"Iteration_{iteration + 1}")
        os.makedirs(iteration_folder, exist_ok=True)  # 为当前 iteration 创建目录

        Different_sample_nums_accuracies.clear()  # 存储不同 sample_nums 的准确率变化

        for sample_num in range(10):
            sample_num += 1
            x_test, y_test = gen_task(train_samples=500 + sample_num)
            test_scale = 1 - (sample_num) / (500 + sample_num)
            x_new_train, x_new_test, y_new_train, y_new_test = train_test_split(x_test, y_test, test_size=test_scale,stratify=y_test)
            weights_before = deepcopy(model.state_dict())

            local_accuracies = []

            # **先进行一次内循环训练（模拟模型适应新任务）**
            for outerstep in range(200):
                loss = train_on_batch(x_new_train, y_new_train)
                if (outerstep + 1) % 1 == 0:
                    predicted = predict(x_new_test)
                    predicted = abs(predicted.round())
                    accuracy = calculate_accuracy(predicted, y_new_test)
                    print(f"Iteration {iteration + 1}, sample_nums: {sample_num}, Outerstep:{outerstep + 1},Accuracy: {accuracy:.3f}")
                    local_accuracies.append((outerstep + 1, accuracy))
            Different_sample_nums_accuracies[sample_num] = local_accuracies
            # **绘制 accuracy vs. outerstep 曲线**
            plt.figure(figsize=(6, 4))
            steps, accs = zip(*local_accuracies)
            plt.plot(steps, accs, marker="o", linestyle="-", label=f"Sample Num: {sample_num}")
            plt.xlabel("Outerstep")
            plt.ylabel("Accuracy")
            plt.title(f"Iteration {iteration + 1}, Sample Num {sample_num}")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(iteration_folder, f"SampleNum_{sample_num}.png"))
            plt.close()

            print(f"Iteration {iteration + 1}, Accuracy: {accuracy:.3f}")
            # print("Predictions:\n", predicted)  # 只打印前 10 个，避免过长
            # print("True values:\n", y_new_test)
            #print("Test data (x_test):\n", x_test[:10])

            model.load_state_dict(weights_before)
        # **保存 CSV 文件**
        csv_path = os.path.join(iteration_folder, "accuracy_data.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Sample_Num", "Outerstep", "Accuracy"])
            for sample_num, acc_list in Different_sample_nums_accuracies.items():
                for outerstep, acc in acc_list:
                    writer.writerow([sample_num, outerstep, acc])

    if Different_sample_nums_accuracies:
        global_accuracies.append(
            np.mean([acc for acc_list in Different_sample_nums_accuracies.values() for _, acc in acc_list]))




# **保存全局准确率到 CSV**
file_path = os.path.join(folder_path, "global_accuracies.csv")
with open(file_path, "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration", "Accuracy"])
    for i, acc in enumerate(global_accuracies, start=1):
        writer.writerow([i * 20, acc])

# **绘制准确率曲线**
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(global_accuracies) + 1), global_accuracies, marker="o")
plt.title("Global Accuracy over Iterations")
plt.xlabel("Iteration (x20)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig(os.path.join(folder_path, "global_accuracy.png"))
plt.show()



