from os import system
import os
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import numpy
import pandas

from sklearn.utils import shuffle,resample


def preprocess(data):
    labelencoder = preprocessing.LabelEncoder()
    not_num = [8,10,11,22]
    #建立虛擬數值
    for i in not_num:
        data["Attribute" + str(i)] = labelencoder.fit_transform(data["Attribute" + str(i)].fillna('0'))
    #填補空缺資料
    for i in range(2,23):
        median = numpy.nanmedian(data["Attribute" + str(i)])
        newData = numpy.where(data["Attribute" + str(i)].isnull(),median,data["Attribute" + str(i)])
        data["Attribute" + str(i)] = newData

    data['Attribute1'] = pandas.to_datetime(data['Attribute1'])
    data['Attribute1'] = pandas.DatetimeIndex(data['Attribute1']).month
    data.dropna()

    data['Attribute22'] = labelencoder.fit_transform(data['Attribute22'] )
    if hasattr(data,'Attribute23'):
        data['Attribute23'] = labelencoder.fit_transform(data['Attribute23'] )

    data = pandas.get_dummies(data)
    return data

def resample_data(data):
    majority = data[data.Attribute23==0]
    minority = data[data.Attribute23==1]

    majority_down_sampled = resample(majority,replace=False,n_samples=3000,random_state=7414)
    data = pandas.concat([majority_down_sampled,minority])
    data = shuffle(data)
    return data

def main():
    train_data = pandas.read_csv("train.csv")
    test_data = pandas.read_csv("test.csv")
    print("Load csv...ok")
    train_data = preprocess(train_data)
    test_data = preprocess(test_data)
    print("Preprocess csv...ok")    
    train_data = resample_data(train_data)
    print("Resample data...ok")
    train_data_output = pandas.DataFrame(train_data["Attribute23"])
    train_data_input = train_data.drop(columns=["Attribute23"],axis=0)
    model = MLPClassifier(solver='adam', activation='logistic',alpha= 0.0001,learning_rate= 'adaptive' , hidden_layer_sizes=(50,50), random_state=1,max_iter=1000,verbose=10,learning_rate_init=0.001)
    model.fit(train_data_input, train_data_output)
    result = model.predict(test_data)
    print("Predict...ok")
    output = pandas.DataFrame(result)
    output.index = output.index.astype(float)
    output.to_csv("submit.csv",header=['ans'],index_label='id')
    print("Output to file...ok")
    os.system("pause")

if __name__ == '__main__':
    main()