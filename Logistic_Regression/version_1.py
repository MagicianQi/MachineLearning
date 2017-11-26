# -*- coding: utf-8 -*-

"""
第一个版本
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


def l(x, y, theta):
    """
    功能：极大似然估计概率函数l(theta)
    参数：数据集，参数
    返回：损失值
    """
    delta = sigmoid(x * theta)
    costs = y.transpose() * np.log(delta) + (1.0 - y).transpose() * np.log(1.0 - delta)
    return sum(costs)


def train(x, y, alpha, maxIter):
    """
    功能：训练函数(变为梯度上升)
    参数：训练集，标签，学习速率，最大迭代次数
    返回：参数
    """
    theta = np.ones((x.shape[1], 1))
    for i in range(maxIter):
        delta = y - sigmoid(x * theta)
        theta = theta + alpha * x.transpose() * delta
        print("Iter:" + str(i+1), "    Alpha:" + str(alpha), "    l:" + str(l(x, y, theta).tolist()[0][0]))
    return theta


def train_SGD(x, y, alpha, maxEpoch):
    """
    功能：训练函数(SGD)
    参数：训练集，标签，学习速率，最大迭代次数
    返回：参数
    """
    theta = np.ones((x.shape[1], 1))
    for i in range(maxEpoch):   # 每次迭代
        for j in range(x.shape[0]): #每个参数
            delta = x[j, :].transpose() * (y[j, :] - sigmoid(x[j, :] * theta))
            theta = theta + alpha * delta
            print("Epoch:" + str(i + 1), "    Iter:" + str(i * maxEpoch + j), "    l:" + str(l(x, y, theta).tolist()[0][0]))
    return theta


def sigmoid(z):
    """
    功能：sigmoid函数
    参数：输入矩阵
    返回：输出矩阵
    """
    return 1.0 / (1 + np.exp(-z))


def get_accuracy(x, y, theta):
    """
    功能：计算测试准确率
    参数：训练集，标签，参数
    返回：准确率
    """
    m = x.shape[0]
    num = 0
    predict_y = sigmoid(x * theta)  # 预测值
    difference = np.abs(y - predict_y).tolist() # 计算预测值与标签的差值的绝对值，小于0.5则预测正确，否则预测错误
    for i in range(m):
        if difference[i][0] < 0.5:
            num += 1
    return num / float(m)


if __name__ == '__main__':
    train_x, train_y = load_data('data.txt')
    pre_train_x, miu, sigma = data_preprocessing(train_x)
    theta = train(train_x, train_y, 0.0001, 100)
    # theta = train_SGD(train_x, train_y, 0.02, 20)
    print(theta)
    print(get_accuracy(pre_train_x, train_y, theta))
