# -*- coding: utf-8 -*-

"""
ML: Linear Regression
Version: 1.0
Author : HyeRi
HomePage : https://www.zhihu.com/people/AraSQ/posts
Email  : qs0727@outlook.com
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
        s_data = line.strip().split(',')  # strip()函数去除换行符，split()函数分割数据
        temp = []
        temp.append(1.0)  # 设置x0 = 1
        if len(s_data) < 2:
            print("输入数据错误")
        for i in range(len(s_data) - 1):
            temp.append(float(s_data[i]))
        x.append(temp)
        y.append(float(s_data[len(s_data) - 1]))
    return np.mat(x), np.mat(y).transpose()  # 转化为矩阵


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
    参数：训练集，学习速率，最大迭代次数
    返回：参数
    """
    theta = np.ones((x.shape[1], 1))
    for i in range(maxIter):
        delta = x.transpose() * (x * theta - y)
        theta = theta - alpha * delta
        print(J(x, y, theta))
    return theta


if __name__ == '__main__':
    train_x, train_y = load_data('data.txt')
    theta = train(train_x, train_y, 0.0000000005, 200)
    print(theta)
