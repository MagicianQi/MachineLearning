# -*- coding: utf-8 -*-

"""
第二个版本：
1.由于学习速率不好设置，所以增加数据预处理，即标准化
2.增加SGD训练
"""

import numpy as np


def load_data(path):
    """
    功能：读取训练数据
    参数：数据文件路径
    返回：训练集与标签矩阵
    """
    x = []
    y = []
    file_in = open(path)
    for line in file_in.readlines():
        s_data = line.strip().split(',')
        temp = []
        temp.append(1.0)
        if len(s_data) < 2:
            print("输入数据错误")
        for i in range(len(s_data) - 1):
            temp.append(float(s_data[i]))
        x.append(temp)
        y.append(float(s_data[len(s_data) - 1]))
    return np.mat(x), np.mat(y).transpose()


def data_preprocessing(x):
    """
    功能：数据预处理
    参数：数据集
    返回：处理后数据，均值，标准差
    """
    m, n = x.shape
    miu = np.zeros((n, 1))
    sigma = np.zeros((n, 1))
    for i in range(n):
        miu[i] = np.mean(x[:, i])
        sigma[i] = np.std(x[:, i])
    for j in range(m):
        x[j, 1:] = np.divide(x[j, 1:] - miu[1:].transpose(), sigma[1:].transpose())
    return x, miu, sigma


def J(x, y, theta):
    """
    功能：损失函数
    参数：数据集，参数
    返回：损失值
    """
    delta = x * theta - y
    ws = 0.5 * delta.transpose() * delta
    return ws


def train(x, y, alpha, maxIter):
    """
    功能：训练函数
    参数：训练集，标签，学习速率，最大迭代次数
    返回：参数
    """
    theta = np.ones((x.shape[1], 1))
    for i in range(maxIter):
        delta = x.transpose() * (x * theta - y)
        theta = theta - alpha * delta
        print("Iter:" + str(i+1), "    J:" + str(J(x, y, theta).tolist()[0][0]))
    return theta


def train_SGD(x, y, alpha, maxEpoch):
    """
    功能：训练函数(SGD)
    参数：训练集，标签，学习速率，最大迭代次数
    返回：参数
    """
    theta = np.ones((x.shape[1], 1))
    for i in range(maxEpoch):
        for j in range(x.shape[0]):
            delta = x[j, :].transpose() * (x[j, :] * theta - y[j, :])
            theta = theta - alpha * delta
            print("Epoch:" + str(i + 1), "    Iter:" + str(i * maxEpoch + j), "    J:" + str(J(x, y, theta).tolist()[0][0]))
    return theta


if __name__ == '__main__':
    train_x, train_y = load_data('data.txt')
    pre_train_x, miu, sigma = data_preprocessing(train_x)
    theta = train(train_x, train_y, 0.02, 20)
    # theta = train_SGD(train_x, train_y, 0.02, 20)
    print(pre_train_x * theta)
