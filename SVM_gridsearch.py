import numpy as np
import pandas as pd
import pylab as pl
from sklearn import svm, cross_validation, grid_search, metrics


CSV_TRAIN = "dataset/train_na2zero.csv"
CSV_TEST = "dataset/test_na2zero.csv"


df_train = pd.read_csv(CSV_TRAIN) #, nrows=600)
Y = df_train.y
X = df_train.iloc[:, 1:].values

# set parameter for grid search
parameters = {'kernel':('rbf','linear'), 'C':[10**i for i in range(-3,3)] , 'gamma':[10**i for i in range(-3,3)]}
clf = svm.SVC()
rbf_svc = grid_search.GridSearchCV(clf, parameters)

# training
n_samples = len(Y)
rbf_svc.fit(X[:n_samples / 2], Y[:n_samples / 2])
print("[best parameter from grid searching]", "\n" , rbf_svc.get_params())

# testing
expected = np.array(Y[n_samples / 2:])
predicted = rbf_svc.predict(X[n_samples / 2:])

# report
print(metrics.classification_report(expected, predicted))
print( metrics.confusion_matrix(expected, predicted))
