# 导入必要的库
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt

# IQ 失衡函数
def iq_imbalance(pha, amp):
    delta_tem = np.random.beta(5, 2)
    delta_tam = delta_tem * (np.pi / 180) * pha
    epsilon_tem = np.random.beta(5, 2)
    epsilon_tam = amp * epsilon_tem
    return delta_tam, epsilon_tam

# 生成 QPSK 信号的符号数据
def message(a, b):
    if a == 2:
        if b % 4 == 0:
            message = np.array([[0, 0]])
        elif b % 4 == 1:
            message = np.array([[0, 1]])
        elif b % 4 == 2:
            message = np.array([[1, 0]])
        elif b % 4 == 3:
            message = np.array([[1, 1]])
        x = 1.0 - 2 * (message[0][0])
        y = 1.0 - 2 * (message[0][1])
        s = (x + 1j * y) * (1.0 / np.sqrt(2))
    return message, s

# 生成含 IQ 失衡的符号数据
def device_symbols(delta, epsilon):
    sym = np.empty(0)
    for i in range(4):
        m, s = message(2, i)
        y_tmp = (1 + epsilon) * np.cos(delta) * s.real + (1 - epsilon) * np.sin(delta) * s.imag + 1j * ((1 - epsilon) * np.cos(delta) * s.imag + (1 + epsilon) * np.sin(delta) * s.real)
        sym = np.append(sym, [y_tmp], axis=0)
    return sym 

# 添加 AWGN 噪声
def add_awgn_noise(symbols, snr_db):
    signal_power = 1.0
    noise_power = signal_power / (10**(snr_db / 10))
    noise_R = np.random.normal(0, np.sqrt(noise_power/2), symbols.shape)
    noise_I = np.random.normal(0, np.sqrt(noise_power/2), symbols.shape)
    noise = noise_R + 1j * noise_I
    return symbols + noise

# 生成象限平均值
def quadrant_means(complex_list):
    complex_array = np.array(complex_list)
    q1 = complex_array[(complex_array.real >= 0) & (complex_array.imag >= 0)]
    q2 = complex_array[(complex_array.real >= 0) & (complex_array.imag < 0)]
    q3 = complex_array[(complex_array.real < 0) & (complex_array.imag >= 0)]
    q4 = complex_array[(complex_array.real < 0) & (complex_array.imag < 0)]
    mean_q1 = np.mean(q1) if len(q1) > 0 else 0
    mean_q2 = np.mean(q2) if len(q2) > 0 else 0
    mean_q3 = np.mean(q3) if len(q3) > 0 else 0
    mean_q4 = np.mean(q4) if len(q4) > 0 else 0
    return np.array([mean_q1.real, mean_q1.imag, mean_q2.real, mean_q2.imag, mean_q3.real, mean_q3.imag, mean_q4.real, mean_q4.imag])

def generate_iq_parameters(N=8, difficulty=0.5):
    """
    生成满足难度约束的IQ失衡参数
    :param N: 设备数量
    :param difficulty: 难度参数（0.0-1.0），0表示差异大/容易识别，1表示差异小/难识别
    :return: delta (相位失衡参数), epsilon (幅度失衡参数)
    """
    max_delta_phase = 35  # 最大相位失衡角度
    max_delta = max_delta_phase * np.pi / 180  # 转换为弧度
    max_epsilon = 0.3  # 最大幅度失衡比例

    # 计算归一化最小距离（difficulty=1时最小间距0，difficulty=0时最小间距0.3）
    min_normalized_distance = (1 - difficulty) * 0.25

    delta = []
    epsilon = []

    # 生成满足距离约束的IQ参数
    for _ in range(N):
        attempts = 0
        while True:
            # 生成随机IQ参数
            delta_tam, epsilon_tam = iq_imbalance(35, 0.3)

            # 归一化到[0,1]范围
            delta_norm = delta_tam / max_delta
            epsilon_norm = epsilon_tam / max_epsilon

            # 检查与已有参数的距离
            valid = True
            for d, e in zip(delta, epsilon):
                d_norm = d / max_delta
                e_norm = e / max_epsilon
                distance = np.sqrt((delta_norm - d_norm) ** 2 + (epsilon_norm - e_norm) ** 2)
                if distance < min_normalized_distance:
                    valid = False
                    break

            if valid or attempts >= 1000:  # 最大尝试次数防止无限循环
                break
            attempts += 1

        delta.append(delta_tam)
        epsilon.append(epsilon_tam)

    return np.array(delta), np.array(epsilon)

# 生成一个 Meta-Learning 任务
def generate_iq_inbalance_data(N=8, train_samples=1000, SNR_train=30,difficulty=1):
    """
    生成一个 Meta-Learning 任务，包含 N 个设备的训练和测试数据
    """
    # 随机生成 IQ 失衡参数


    # delta = np.empty((0, 1))
    # epsilon = np.empty((0, 1))
    # for _ in range(N):
    #     delta_tam, epsilon_tam = iq_imbalance(35, 0.3)
    #     delta = np.append(delta, delta_tam)
    #     epsilon = np.append(epsilon, epsilon_tam)

    delta, epsilon= generate_iq_parameters(N, difficulty)
    # 训练数据
    train_data = np.empty((0, 8))
    train_labels = np.empty(0)


    # 测试数据
    test_data = np.empty((0, 8))
    test_labels = np.empty(0)

    # 生成每个设备的数据
    for i in range(N):
        symbols_list = device_symbols(delta[i], epsilon[i])

        # 生成训练样本
        for _ in range(train_samples):
            noisy_symbols = add_awgn_noise(symbols_list, SNR_train)
            feature_vector = quadrant_means(noisy_symbols)
            train_data = np.append(train_data, [feature_vector], axis=0)
            train_labels = np.append(train_labels, [i])

    return (train_data, train_labels)


# 测试 Meta-Learning 任务生成
(train_data, train_labels) = generate_iq_inbalance_data(train_samples=10)

# print("train_data:\n", train_data[:100])  # 只打印前 10 个，避免过长
# print("train_labels:\n", train_labels[:100])
