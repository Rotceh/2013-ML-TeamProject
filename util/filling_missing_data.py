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
    df_part = pd.read_csv(CSV_TRAIN, nrows=1200).fillna(0)
    grouped = df_part.groupby("y")
    print([add_missing_data_with_group_mean(one_group_label,one_group) for
           one_group_label,one_group in grouped ])
