import math
import numpy as np
import time
class Adaboost:
    stepsize=10
    def __init__(self,stepsize):
        self.stepsize=stepsize

    def I(self,pred, true):  # I(x)
        if pred != true:
            return 1
        else:
            return 0

    def losscalculate(self,predict, weight, label):  # em计算函数
        loss = 0
        if len(predict) != len(weight):
            raise IndexError
        for pred, wgt, lb in zip(predict, weight, label):
            loss += wgt * self.I(pred, lb)
        return loss  # 计算弱分类器loss

    def DecisionStumppredict(self,data, key):  # Gm(x)预测
        loss = 0
        predict = []
        for i in range(len(data)):
            if data[i] < key:
                predict.append(-1)
            else:
                predict.append(1)
        return np.array(predict)  # 弱分类器预测

    def buildStump(self,dataMatrix, weightmatrix, classLabels, stepSize=30):#构建弱分类器
        m = dataMatrix.shape[0]
        bestStump = []
        bestError = 0

        bestkey = 0
        minError = float("inf")  # 最小错误率初始化为无穷大
        rangeMin = dataMatrix.min();
        rangeMax = dataMatrix.max();
        step = (rangeMax - rangeMin) / stepSize#逐项搜索最优割点
        for j in np.arange(rangeMin, rangeMax, step):
            loss = self.losscalculate(self.DecisionStumppredict(dataMatrix, j), weightmatrix, classLabels)
            if loss < minError:
                bestError = loss
                bestkey = j
        bestStump.append(bestError)
        bestStump.append(bestkey)
        return np.array(bestStump)  # 构建基于不同特征的弱分类器

    def alpha(self,loss):#计算alpha
        return 1 / 2 * math.log((1 - loss) / loss)

    def calZ(self,Alpha, label, predict, weight):
        Z = 0
        for i, k, l in zip(label, weight, predict):
            Z += k * math.exp(-Alpha * i * l)
        return Z

    def Adaboost(self,data, label,stepsize):#对每个特征进行训练
        stumps = []
        Alphas = []
        weight = np.zeros(data.shape[0])
        weight.fill(1 / data.shape[0])
        start = time.clock()
        for w in range(data.shape[1]):

            stump = self.buildStump(data[:, w], weight, label,stepsize)
            Alpha = self.alpha(stump[0])
            predict = self.DecisionStumppredict(data[:, w], stump[1])#弱分类器预测函数
            Z = self.calZ(Alpha, label, predict, weight)
            cnt = 0
            for i, j, k in zip(weight, label, predict):
                weight[cnt] = i * math.exp(-Alpha * j * k)/Z#权重更新
                cnt += 1
            stumps.append(stump)
            Alphas.append(Alpha)
            print("training completed:{:.2f}%".format((w + 1) / data.shape[1] * 100) + " ETA={:.2f}s".format(
                (time.clock() - start) * (data.shape[1] - w - 1) / (w + 1)))
        return stumps, Alphas

    def predict(self,data, stumps, alphas):
        Pred = np.zeros(data.shape[0])

        for i in range(data.shape[1]):
            cnt = 0
            pred = self.DecisionStumppredict(data[:, i], stumps[i][1])
            for j, k in zip(pred, Pred):
                Pred[cnt] += j * alphas[i]#加权累计各个弱分类器的预测结果
                cnt += 1
        cnt = 0
        for i in range(len(Pred)):
            Pred[i] = np.sign(Pred[i])
        return Pred