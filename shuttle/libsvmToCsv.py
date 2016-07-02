from sklearn.datasets import load_svmlight_file
import numpy as np
import pandas as pd


def get_data(file):
    data = load_svmlight_file(file)
    X = pd.DataFrame(data[0].todense())
    y = pd.DataFrame(data[1],columns=['class'])
    return X,y

train,train_targets = get_data('shuttle.scale')
train['class'] = train_targets['class']

test,test_targets = get_data('shuttletest.scale')
test['class'] = test_targets['class']

train.to_csv('shuttle_train.csv', index=False)
test.to_csv('shuttle_test.csv', index=False)
