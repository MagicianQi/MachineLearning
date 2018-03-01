# -*- coding: utf-8 -*-

import numpy as np

import random


class SVM:
    def __init__(self, train_x, train_y, C, thresh, kernelOption):
        self.train_x = train_x  # 训练样本
        self.train_y = train_y  # 标签
        self.C = C  # 惩罚因子 惩罚因子越大越不希望看到离群点
        self.thresh = thresh  # 迭代终止条件
        self.numSamples = train_x.shape[0]  # 训练样本数目
        self.alpha = np.mat(np.zeros((self.numSamples, 1)))  # 所有训练样本的拉格朗日因子
        self.b = 0
        self.kernelOption = kernelOption   # 核参数
        self.alphasIndex = []
        self.kernelMatrix = get_kernel_matrix(self.train_x, self.kernelOption)     # 核矩阵K


def load_data(path):
    """
    读取训练数据
    :param path: 数据文件路径
    :return: 训练集与标签矩阵
    """
    x = []
    y = []
    file_in = open(path)
    for line in file_in.readlines():
        s_data = line.strip().split(',')  # strip()函数去除换行符，split()函数分割数据
        temp = []
        if len(s_data) < 2:
            print("输入数据错误")
        for i in range(len(s_data) - 1):
            temp.append(float(s_data[i]))
        x.append(temp)
        y.append(float(s_data[len(s_data) - 1]))
    return np.mat(x), np.mat(y).transpose()  # 转化为矩阵


def kernel(vector_x, vector_z, kernelOption):
    """
    核函数计算
    :param vector_x: 向量 x
    :param vector_z: 向量 z
    :param kernelOption: 核参数
    :return: 核函数结果
    """
    kernelType = kernelOption[0]
    if kernelType == 'polynomial':           # 多项式核 d为幂次数
        d = kernelOption[1]
        kernelValue = np.power(vector_x * vector_z.T, d)
    elif kernelType == 'rbf':         # 高斯核 sigma为波长
        sigma = kernelOption[1]
        if sigma == 0:
            sigma = 1.0
        kernelValue = np.mat(np.zeros((vector_x.shape[0], 1)))
        for i in range(vector_x.shape[0]):
            diff = vector_x[i, :] - vector_z
            kernelValue[i] = np.exp(diff * diff.T / (-2.0 * sigma ** 2))
    else:
        raise NameError('Only Support Polynomial and Gaussian Kernel Type!')
    return kernelValue


def get_kernel_matrix(matrix_x, kernelOption):
    """
    核矩阵计算
    :param matrix_x: 输入特征矩阵
    :param kernelOption: 核参数
    :return: 核矩阵
    """
    m = matrix_x.shape[0]
    kernelMatrix = np.mat(np.zeros((m, m)))
    for i in range(m):
        for j in range(m):
            kernelMatrix[i, j] = kernel(matrix_x[i, :], matrix_x[j, :], kernelOption)
    return kernelMatrix


def update_alpha(svm, type):
    """
    更新一次alpha 即一组alpha
    :return: 超平面参数 b
    """

    if type == 'All':
        svm.alphasIndex = [k for k in range(svm.train_x.shape[0])]
    elif type == 'SupportVector':
        svm.alphasIndex = np.nonzero((svm.alpha.A > 0) * (svm.alpha.A < svm.C))[0]
    else:
        svm.alphasIndex = [k for k in range(svm.train_x.shape[0])]
    count = 0       # 更新alpha的组数
    for i in svm.alphasIndex:
        if not KKT_condition_judgment(svm, i):      # 此处没有选择违反KKT条件最大的alpha 按顺序挑选
            error_i = get_error(svm, i)
            j = select_alpha_j(svm, i)
            error_j = get_error(svm, j)
            alpha_i_old = svm.alpha[i].copy()
            alpha_j_old = svm.alpha[j].copy()
            if svm.train_y[i] != svm.train_y[j]:        # 计算边界 L与H
                L = max(0, svm.alpha[j] - svm.alpha[i])
                H = min(svm.C, svm.C + svm.alpha[j] - svm.alpha[i])
            else:
                L = max(0, svm.alpha[j] + svm.alpha[i] - svm.C)
                H = min(svm.C, svm.alpha[j] + svm.alpha[i])
            if L == H:
                continue
            # 判断样本 i 与 j的相似度
            eta = 2.0 * kernel(svm.train_x[i, :], svm.train_x[j, :], svm.kernelOption) - kernel(svm.train_x[i, :], svm.train_x[i, :], svm.kernelOption) \
                    - kernel(svm.train_x[j, :], svm.train_x[j, :], svm.kernelOption)
            if eta >= 0:
                continue
            svm.alpha[j] -= svm.train_y[j] * (error_i - error_j) / eta
            if svm.alpha[j] > H:
                svm.alpha[j] = H
            if svm.alpha[j] < L:
                svm.alpha[j] = L
            if abs(alpha_j_old - svm.alpha[j]) < 0.00001:
                continue
            svm.alpha[i] += svm.train_y[i] * svm.train_y[j] * (alpha_j_old - svm.alpha[j])      # 更新alpha i
            b1 = svm.b - error_i - svm.train_y[i] * (svm.alpha[i] - alpha_i_old) * kernel(svm.train_x[i, :], svm.train_x[i, :], svm.kernelOption) \
                 - svm.train_y[j] * (svm.alpha[j] - alpha_j_old) * kernel(svm.train_x[i, :], svm.train_x[j, :], svm.kernelOption)
            b2 = svm.b - error_j - svm.train_y[i] * (svm.alpha[i] - alpha_i_old) * kernel(svm.train_x[i, :], svm.train_x[j, :], svm.kernelOption) \
                 - svm.train_y[j] * (svm.alpha[j] - alpha_j_old) * kernel(svm.train_x[j, :], svm.train_x[j, :], svm.kernelOption)
            if (0 < svm.alpha[i]) and (svm.alpha[i] < svm.C):   # 如果 i 是支持向量
                svm.b = b1
            elif (0 < svm.alpha[j]) and (svm.alpha[j] < svm.C):
                svm.b = b2
            else:
                svm.b = (b1 + b2) / 2.0
            count += 1
    return svm.b, count


def select_alpha_j(svm, i):
    """
    选择SMO算法的第二个alpha
    :return: 第 j 个alpha
    """
    while 1:
        if len(svm.alphasIndex) != 0:
            j = svm.alphasIndex[random.randint(0, len(svm.alphasIndex) - 1)]   # 此处也没有选择|Ei - Ej| 最大的 alpha j , 而是随机挑选
            if i != j:
                return j
        else:
            j = int(random.uniform(0, svm.numSamples))
            return j


def get_error(svm, k):
    """
    获取第k个样本的误差
    :return: 误差值
    """
    # kernel_k = get_kernel_matrix(train_x, kernelOption)[:, k]
    kernel_k = svm.kernelMatrix[:, k]
    output_k = float(np.multiply(svm.alpha, svm.train_y).T * kernel_k + svm.b)
    error_k = output_k - float(svm.train_y[k])
    return error_k


def get_func_distance(svm, k):
    """
    获取第k个样本的函数距离
    :return: 函数距离
    """
    # kernel_k = get_kernel_matrix(train_x, kernelOption)[:, k]   # 根据公式计算
    kernel_k = svm.kernelMatrix[:, k]
    output_k = float(np.multiply(svm.alpha, svm.train_y).T * kernel_k + svm.b)
    func_distance = output_k * svm.train_y[k]
    return func_distance


def KKT_condition_judgment(svm, k):
    """
    判断第k个样本点对应的alpha是否符合KKT条件
    :return:判断结果
    """
    # 函数距离小于1，那么alpha必须等于。函数距离大于1，那么alpha必须要等于0。否则就违反了KKT条件。
    if (get_func_distance(svm, k) - 1 < -svm.thresh) and (svm.alpha[k] < svm.C) or \
                    (get_func_distance(svm, k) - 1 > svm.thresh) and (svm.alpha[k] > 0):
        return 0
    else:
        return 1


def KKT_condition(svm, typeOption='All'):
    """
    判断所有样本对应的的alpha 或者 支持向量对应的 alpha 是否全部符合 KKT 条件
    :return: 判断结果
    """
    if typeOption == 'All':     # 判断所有alpha
        for i in range(svm.train_x.shape[0]):
            if not KKT_condition_judgment(svm, i):
                return 0
    elif typeOption == 'SupportVector':     # 只判断支持向量对应的alpha
        for j in range(svm.train_x.shape[0]):
            if svm.alpha[j] != 0 and svm.alpha[j] != svm.C:
                if not KKT_condition_judgment(svm, j):
                    return 0
    else:
        raise NameError('Only Support All , SupportVector And Single Type!')
    return 1


def train(train_x, train_y, C, thresh, kernelOption=('G', 1.0)):
    """
    训练函数
    :param train_x: 训练集
    :param train_y: 标签
    :param C: 惩罚因子
    :param thresh: KKT条件判断阈值
    :param kernelOption: 核参数
    :return: alpha 和 b
    """
    svm = SVM(train_x, train_y, C, thresh, kernelOption)
    iterCount = 0
    while not KKT_condition(svm, 'All'):
        iterCount += 1
        b, revise_count = update_alpha(svm, 'All')
        print('Iter : ' + str(iterCount) + '\t' + 'All' + '\t' + str(revise_count))
        if revise_count == 0:
            return svm
        while not KKT_condition(svm, 'SupportVector'):
            iterCount += 1
            b, revise_count = update_alpha(svm, 'SupportVector')
            print('Iter : ' + str(iterCount) + '\t' + 'Alpha: 0-C' + '\t' + str(revise_count))
            if revise_count == 0:
                break
    return svm


def predict(svm, test_x, test_y):
    """
    测试函数
    :param svm: svm结构
    :param test_x: 测试集
    :param test_y: 测试集标签
    :return: 预测准确率
    """
    test_x = np.mat(test_x)
    test_y = np.mat(test_y)
    numTestSamples = test_x.shape[0]    # 样本个数
    supportVectorsIndex = np.nonzero((svm.alpha.A > 0) * (svm.alpha.A < svm.C))[0]   # 支持向量索引
    supportVectors = svm.train_x[supportVectorsIndex]
    supportVectorLabels = svm.train_y[supportVectorsIndex]
    supportVectorAlphas = svm.alpha[supportVectorsIndex]
    matchCount = 0      # 预测正确个数
    for i in range(numTestSamples):
        kernelValue = kernel(supportVectors, test_x[i, :], svm.kernelOption)
        predict = kernelValue.T * np.multiply(supportVectorLabels, supportVectorAlphas) + svm.b
        if np.sign(predict) == np.sign(test_y[i]):
            matchCount += 1
    accuracy = float(matchCount) / numTestSamples
    return accuracy


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


if __name__ == '__main__':
    x, y = load_data('data.txt')
    x, _, _ = data_preprocessing(x)
    svm = train(x[0:80, :], y[0:80, :], 0.6, 0.001, ('polynomial', 1))
    print(predict(svm, x[80:100, :], y[80:100, :]))
