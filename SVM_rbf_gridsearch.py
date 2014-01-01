# import numpy as np
import pandas as pd
# import pylab as pl
from sklearn import svm, grid_search, metrics
from sklearn.cross_validation import train_test_split


CSV_TRAIN = "dataset/train_zero_60x60.csv"
CSV_TEST = "dataset/test_zero_60x60.csv"

GRID_RESULT = "results/grid_CV/rbf_SVM_grid_60x60_0101.txt"

df_train = pd.read_csv(CSV_TRAIN)  #, nrows=1000)
Y = df_train.y
X = df_train.iloc[:, 1:].values

# set parameter for grid search
param_list  = [
    {'kernel': ['rbf'],
     'C': [1, 10, 50,
           100, 200, 500,
           1000],
     'gamma': [1, 0.5, 0.2,
               0.1, 0.05, 0.02,
               0.01, 0.005, 0.002,
               0.001, 0.0005, 0.0002,
               0.0001, 0.00005, 0.00002,
               0.00001]},
]

svm_clf = svm.SVC()
lin_clf = svm.LinearSVC()
# poly_clf = svm.SVC()

train_X, test_X, train_Y, test_Y = train_test_split(
    X, Y, test_size=0.25, random_state=5
)

for clf, param in zip((svm_clf, lin_clf), param_list):
    grid_cv = grid_search.GridSearchCV(clf, param, n_jobs=16, verbose=1)

    # training
    grid_cv.fit(train_X, train_Y)

    # print best estimator
    with open(GRID_RESULT, "a") as f:
        msg = (
            "\n\n\n>>>> [best parameter from grid searching] <<<<\n",
            grid_cv.best_estimator_,
            "\n\n\n"
        )
        print(*msg)
        print(*msg, file=f)
        for params, mean_score, scores in grid_cv.grid_scores_:
            row_msg = (
                "%0.3f (+/-%0.03f) for %r"
                % (mean_score, scores.std() / 2, params)
            )
            print(row_msg)
            print(row_msg, file=f)

    # testing
    pred_Y = grid_cv.predict(test_X)
    # report
    print(metrics.classification_report(test_Y, pred_Y))
    print(metrics.confusion_matrix(test_Y, pred_Y))
