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


def get_classLoss(train_x, imp_label, center):
    """
    获取每一🀄️种分类损失值
    :param train_x: 训练集
    :param imp_label: 训练集的聚类标签
    :param center: 该分类的质心矩阵
    :return: 损失值
    """
    loss = 0
    for i in range(train_x.shape[0]):
        temp = 0
        for j in range(train_x.shape[1]):
            temp += (train_x[i, j] - center[int(imp_label[i]), j]) ** 2
        loss += temp
    return loss


def get_elbow_position(classLoss, times):
    """
    寻找 损失值与分类个数构造的函数的拐点。
    先计算每一种分类损失值与后一种分类损失值的差值 dif，
    然后计算每一个差值与后一个差值的倍数，若大与times，则认为是拐点。
    :param classLoss: 每一种分类的损失值列表
    :param times: 倍数阈值
    :return: 
    """
    if len(classLoss) < 3:
        return classLoss.index(min(classLoss))
    dif = []
    for i in range(len(classLoss) - 1):
        dif.append(classLoss[i] - classLoss[i + 1])
    for j in range(len(dif) - 1):
        if dif[j] / dif[j + 1] > times:
            return j + 1
    return len(classLoss) - 1


def train(train_x, maxIter, maxClassNum):

    """
    训练函数
    :param train_x: 训练样本
    :param maxIter: 最大迭代次数
    :param classNum: 最大类别个数
    :return:
    """
    numSamples = train_x.shape[0]                       # 样本个数
    imp_label = []                                      # 样本聚类标签
    for i in range(maxClassNum):
        imp_label.append(np.zeros((numSamples, 1)))
    center = [np.zeros((classNum, train_x.shape[1])) for classNum in range(2, maxClassNum + 1)]
    for i in range(len(center)):                        # 初始化所有质心
        initCenter(center[i], train_x)
    classLoss = [0 for i in range(maxClassNum - 1)]         # 初始化每一类的损失值
    for iter in range(maxIter):                                     # 每一次迭代
        print('Iter ' + str(iter) + ':')
        for eachClass in range(len(center)):                        # 每一种分类
            for i in range(numSamples):                             # 先迭代样本
                cost = J(train_x[i, :], center[eachClass])          # 计算该样本与所有质心距离
                imp_label[eachClass][i] = cost.index(min(cost))     # 距离最小的质心的索引作为聚类标签
            for j in range(center[eachClass].shape[0]):             # 迭代质心
                for k in range(center[eachClass].shape[1]):
                    temp = [train_x[m, k] for m in np.where(imp_label[eachClass][:] == j)[0]]  # 标签为 j 的所有样本的集合
                    if float(len(temp)) != 0:
                        center[eachClass][j, k] = sum(temp) / float(len(temp))  # 求平均值
            print('Class ' + str(eachClass + 2) + ' : ')
            print([len([m for m in np.where(imp_label[eachClass][:] == t)[0]]) for t in range(eachClass + 2)])  # 每一类的个数
            classLoss[eachClass] = get_classLoss(train_x, imp_label[eachClass], center[eachClass])
    print('Each class loss : ')
    print(classLoss)
    elbow_class = get_elbow_position(classLoss, 2)          # 求损失值与分类个数构造的函数的 拐点
    print('Select classification ： ' + str(elbow_class + 2))
    print([len([m for m in np.where(imp_label[elbow_class][:] == t)[0]]) for t in range(elbow_class + 2)])


if __name__ == '__main__':
    x = load_data('data.txt')
    train(x, 10, 10)
