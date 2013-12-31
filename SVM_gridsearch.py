import numpy as np
import pandas as pd
# import pylab as pl
from sklearn import svm, grid_search, metrics
from sklearn.cross_validation import train_test_split


CSV_TRAIN = "dataset/train_na2zero.csv"
CSV_TEST = "dataset/test_na2zero.csv"


df_train = pd.read_csv(CSV_TRAIN)   # nrows=3000)
Y = df_train.y
X = df_train.iloc[:, 1:].values

# set parameter for grid search
parameters = {
    'kernel': 'rbf',
    'C': [10 ** i for i in range(-3, 3)],
    'gamma': [10 ** i for i in range(-6, 2)]
}
rbf_clf = svm.SVC()
lin_clf = svm.LinearSVC()

train_X, test_X, train_Y, test_Y = train_test_split(
    X, Y, test_size=0.25, random_state=5
)

for clf in (rbf_clf, lin_clf):
    grid_cv = grid_search.GridSearchCV(clf, parameters, n_jobs=12, verbose=2)
    # training
    grid_cv.fit(train_X, train_Y)
    # print best estimator
    print(
        "[best parameter from grid searching]\n",
        grid_cv.best_estimator_
    )
    for params, mean_score, scores in clf.grid_scores_:
        print(
            "%0.3f (+/-%0.03f) for %r"
            % (mean_score, scores.std() // 2, params)
        )

# testing
# expected = np.array(Y[n_samples / 2:])
# predicted = rbf_svc.predict(X[n_samples / 2:])

# report
# print(metrics.classification_report(expected, predicted))
# print( metrics.confusion_matrix(expected, predicted))
