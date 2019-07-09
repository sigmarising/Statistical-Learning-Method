from sklearn.preprocessing import LabelEncoder
import pandas as pd
data=pd.read_csv("adult.csv",header=None)
data.columns=["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex"
    ,"capital-gain" ,"capital-loss","hours-per-week","native-country","income"]
del data["fnlwgt"]
education_num=data["education-num"]
native=data["native-country"]
data=data.drop("native-country",axis=1)
data=data.drop("education-num",axis=1)
data.insert(1,"education_num",education_num)
data.insert(7,"native-country",native)
value=data.values
for i in value:
    if i[-1]==" <=50K":
        i[-1]=-1
    else:
        i[-1] = 1
scaler=LabelEncoder()
for i in range(2,10):
    value[:,i]=scaler.fit_transform(value[:,i])
train=open("train.csv","w")
test=open("test.csv","w")
cnt=0
for i in value:
    if cnt<27000:
        train.write(",".join(str(w)for w in i))
        train.write("\n")
    else:
        test.write(",".join(str(w) for w in i))
        test.write("\n")
    cnt+=1
train.close()
test.close()


