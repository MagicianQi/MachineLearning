# -*- coding: utf-8 -*-

import numpy as np
import random
import math


def load_data(gauss_1, gauss_2, num=100):
    """
    随机生成两个高斯分布的数据
    :param gauss_1: 第一个高斯分布参数
    :param gauss_2: 第二个高斯分布参数
    :param num: 生成数据个数
    :return: 生成的数据集
    """
    x = []
    while len(x) < num:
        if random.random() > 0.5:
            x.append(random.gauss(gauss_1[0], gauss_1[1]))
        else:
            x.append(random.gauss(gauss_2[0], gauss_2[1]))
    return np.array(x)


def E(train_x, w, miu, sigma):
    """
    EM算法中的E步，主要通过已知的混合高斯分布参数(miu, sigma)来计算每一个样本属于哪一个高斯分布的概率(w)
    :param train_x: 训练集
    :param w: 每个样本属于每个每一类的概率
    :param miu: 高斯分布参数 miu
    :param sigma: 高斯分布参数 sigma
    :return:
    """
    for i in range(train_x.shape[0]):
        total = 0.0
        for k in range(miu.shape[0]):
            total += math.exp((-1 * (float(train_x[i] - miu[k])) ** 2 / (2 * (float(sigma[k] ** 2)))))
        for j in range(miu.shape[0]):
            if total:
                w[i, j] = math.exp((-1 * (float(train_x[i] - miu[j])) ** 2 / (2 * (float(sigma[j] ** 2))))) / total
            else:
                w[i, j] = 0


def M(train_x, w, miu, sigma):
    """
    EM算法中的M步，主要通过已知的样本属于每一类的概率(w),利用极大似然估计求每一个高斯分布的参数(miu, sigma)
    :param train_x:
    :param w: 每个样本属于每个每一类的概率
    :param miu: 高斯分布参数 miu
    :param sigma: 高斯分布参数 sigma
    :return:
    """
    for j in range(miu.shape[0]):
        miu[j] = np.sum(np.dot(w[:, j], train_x)) / float(np.sum(w[:, j]))
    for i in range(sigma.shape[0]):
        sigma[i] = math.sqrt(float(np.sum(np.dot(w[:, i], (train_x - miu[i]) ** 2))) / float(np.sum(w[:, i])))


def train(train_x, maxIter, classNum):
    """
    训练函数
    :param train_x: 训练集
    :param maxIter: 最大迭代次数
    :param classNum: 混合高斯分布个数
    :return:
    """
    m = train_x.shape[0]
    w = np.zeros((m, classNum))
    miu = np.random.rand(classNum)
    sigma = np.random.rand(classNum) + 3
    for iter in range(maxIter):
        E(train_x, w, miu, sigma)
        M(train_x, w, miu, sigma)
        print('Iter' + str(iter) + ' : ', end='')
        print(miu, end='')
        print(sigma)


if __name__ == '__main__':
    x = load_data([-10, 3], [10, 3], 100)
    train(x, 100, 2)
