from sklearn.tree import DecisionTreeRegressor
import numpy as np
import scipy.optimize as opt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
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
GBDT=GradientBoostingClassifier()
GBDT.fit(traindata,trainlabel)
predict=GBDT.predict(testdata)
print("acc={}".format(acc(predict,testlabel)))