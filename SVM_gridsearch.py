# import numpy as np
import pandas as pd
# import pylab as pl
from sklearn import svm, grid_search, metrics
from sklearn.cross_validation import train_test_split


CSV_TRAIN = "dataset/train_na2zero.csv"
CSV_TEST = "dataset/test_na2zero.csv"


df_train = pd.read_csv(CSV_TRAIN)  #, nrows=1000)
Y = df_train.y
X = df_train.iloc[:, 1:].values

# set parameter for grid search
parameters = {
    'kernel': ['poly'],
    'C': [10 ** i for i in range(-7, 5)],
    # 'gamma': [10 ** i for i in range(-5, -2)]
    'degree': [i for i in range(2, 6)]
}
rbf_clf = svm.SVC()
lin_clf = svm.LinearSVC()
poly_clf = svm.SVC()

train_X, test_X, train_Y, test_Y = train_test_split(
    X, Y, test_size=0.25, random_state=5
)

grid_cv = grid_search.GridSearchCV(poly_clf, parameters, n_jobs=14, verbose=2)

# training
grid_cv.fit(train_X, train_Y)
# print best estimator

print(
    "\n\n\n>>>> [best parameter from grid searching] <<<<\n",
    grid_cv.best_estimator_,
    "\n\n\n"
)
with open("poly_SVM_grid_result.txt", "w") as f:
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
