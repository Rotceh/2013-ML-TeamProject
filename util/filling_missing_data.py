'''
Created on Jan 1, 2014

@author: c3h3
'''

import numpy as np


def add_missing_data_with_group_mean(one_group_label, one_group_df):
    one_group_data_array = one_group_df.values[:,1:]
    one_group_mean = np.mean(one_group_data_array,axis=0)
    one_group_new_data_array = np.apply_along_axis(lambda one_row: one_row + one_group_mean, 1, one_group_data_array)
    N_ROW = one_group_new_data_array.shape[0]
    return np.c_[one_group_label*np.ones(N_ROW), one_group_new_data_array]
    #return one_group_new_data_array



if __name__ == '__main__':
    pass
    
    import pandas as pd
    
    CSV_TRAIN = "../dataset/train_zero_60x60.csv"
    df_part = pd.read_csv(CSV_TRAIN, nrows=6000).fillna(0)
    grouped = df_part.groupby("y")
    _filled_data = [add_missing_data_with_group_mean(one_group_label,one_group) for one_group_label,one_group in grouped ]

    filled_data = _filled_data[0]

    for one_data in _filled_data[1:]:
        filled_data = np.r_[filled_data, one_data]

    print filled_data

    X = filled_data[:,1:]
    Y = filled_data[:,0]

    print "Y = ",Y
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    from scipy.ndimage import convolve
    from sklearn import linear_model, datasets, metrics
    from sklearn.cross_validation import train_test_split
    from sklearn.neural_network import BernoulliRBM
    from sklearn.pipeline import Pipeline
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=0)
    
    # Models we will use
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)
    
    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    
    ###############################################################################
    # Training
    
    # Hyper-parameters. These were set by cross-validation,
    # using a GridSearchCV. Here we are not performing cross-validation to
    # save time.
    rbm.learning_rate = 0.06
    rbm.n_iter = 10
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = 144
    logistic.C = 6000.0
    
    # Training RBM-Logistic Pipeline
    classifier.fit(X_train, Y_train)
    
    # Training Logistic regression
    logistic_classifier = linear_model.LogisticRegression(C=100.0)
    logistic_classifier.fit(X_train, Y_train)
    
    ###############################################################################
    # Evaluation
    
    print()
    print("Logistic regression using RBM features:\n%s\n" % (
        metrics.classification_report(
            Y_test,
            classifier.predict(X_test))))
    
    print("Logistic regression using raw pixel features:\n%s\n" % (
        metrics.classification_report(
            Y_test,
            logistic_classifier.predict(X_test))))
    

    ###############################################################################
    # Plotting
    
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(rbm.components_):
        plt.subplot(12, 12, i + 1)
        plt.imshow(comp.reshape((60, 60)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('100 components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    
    plt.show()
    
    

#import numpy as np
#import matplotlib.pyplot as plt
#
#from scipy.ndimage import convolve
#from sklearn import linear_model, datasets, metrics
#from sklearn.cross_validation import train_test_split
#from sklearn.neural_network import BernoulliRBM
#from sklearn.pipeline import Pipeline
#
#
################################################################################
## Setting up
#
#def nudge_dataset(X, Y):
#    """
#    This produces a dataset 5 times bigger than the original one,
#    by moving the 8x8 images in X around by 1px to left, right, down, up
#    """
#    direction_vectors = [
#        [[0, 1, 0],
#         [0, 0, 0],
#         [0, 0, 0]],
#
#        [[0, 0, 0],
#         [1, 0, 0],
#         [0, 0, 0]],
#
#        [[0, 0, 0],
#         [0, 0, 1],
#         [0, 0, 0]],
#
#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 1, 0]]]
#
#    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
#                                  weights=w).ravel()
#    X = np.concatenate([X] +
#                       [np.apply_along_axis(shift, 1, X, vector)
#                        for vector in direction_vectors])
#    Y = np.concatenate([Y for _ in range(5)], axis=0)
#    return X, Y
#
## Load Data
#digits = datasets.load_digits()
#X = np.asarray(digits.data, 'float32')
#X, Y = nudge_dataset(X, digits.target)
#
#print list(Y) 
    
    
    
    
    