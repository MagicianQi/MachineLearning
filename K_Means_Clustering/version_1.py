# -*- coding: utf-8 -*-

import numpy as np

import random


def load_data(path):
    """
    读取数据
    :param path: 数据文件路径
    :return: 数据集
    """
    x = []
    file_in = open(path)
    for line in file_in.readlines():
        s_data = line.strip().split('   ')  # strip()函数去除换行符，split()函数分割数据
        if len(s_data) < 2:
            print("输入数据错误")
        temp = []
        for i in range(len(s_data)):
            temp.append(float(s_data[i]))
        x.append(temp)
    return np.mat(x)  # 转化为矩阵


def initCenter(center, train_x):
    """
    初始化各个类的质心，在每一维最大值与最小值之前取随机值初始化，实际并不需要，实数范围内都可以。
    :param center: 质心矩阵
    :param train_x: 数据集
    :return:
    """
    min = [np.min(train_x[:, i]) for i in range(train_x.shape[1])]
    max = [np.max(train_x[:, i]) for i in range(train_x.shape[1])]
    for i in range(center.shape[0]):
        for j in range(center.shape[1]):
            center[i, j] = (random.uniform(-1, 1) * (max[j] - min[j]) / 2.0) + ((max[j] + min[j]) / 2.0)


def J(sample, center):
    """
    计算一个样本与所有质心的距离
    :param sample: 样本
    :param center: 质心矩阵
    :return: 与所有质心距离列表
    """
    cost = []
    for k in range(center.shape[0]):
        temp = 0
        for i in range(center.shape[1]):
            temp += (sample[0, i] - center[k, i]) ** 2
        cost.append(temp)
    return cost


def train(train_x, maxIter, classNum):
    """
    训练函数
    :param train_x: 训练样本
    :param maxIter: 最大迭代次数
    :param classNum: 类别个数
    :return:
    """
    numSamples = train_x.shape[0]           # 样本个数
    imp_label = np.zeros((numSamples, 1))   # 样本聚类标签
    center = np.zeros((classNum, train_x.shape[1]))     # 每一类质心初始化
    initCenter(center, train_x)                         # 初始化质心
    for iter in range(maxIter):                         # 每一次迭代
        for i in range(numSamples):                     # 先迭代样本
            cost = J(train_x[i, :], center)             # 计算该样本与所有质心距离
            imp_label[i] = cost.index(min(cost))        # 距离最小的质心的索引作为聚类标签
        for j in range(center.shape[0]):                # 迭代质心
            for k in range(center.shape[1]):
                temp = [train_x[m, k] for m in np.where(imp_label[:] == j)[0]]  # 标签为 j 的所有样本的集合
                if float(len(temp)) != 0:
                    center[j, k] = sum(temp) / float(len(temp))     # 求平均值
        print('Iter ' + str(iter) + ':')
        print([len([m for m in np.where(imp_label[:] == t)[0]]) for t in range(classNum)])  # 每一类的个数


if __name__ == '__main__':
    x = load_data('data.txt')
    train(x, 10, 4)
