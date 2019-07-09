import pandas as pd
from Adaboost import Adaboost
def acc(pred,label):
    cor=0
    for i,j in zip(pred,label):
        if i==int(j):
            cor+=1
    return cor/len(label)
data=pd.read_csv("train.csv",header=None)
traindata=data.values[:,:-1]
trainlabel=data.values[: ,-1]
test=pd.read_csv("test.csv",header=None)
testdata=data.values[:,:-1]
testlabel=data.values[:,-1]
stepsize=20#弱分类器最优割点搜索步长
ada=Adaboost(stepsize)
stumps,Alphas=ada.Adaboost(traindata,trainlabel,stepsize)
Predict=ada.predict(testdata,stumps,Alphas)
print("acc={}".format(acc(Predict,testlabel)))


