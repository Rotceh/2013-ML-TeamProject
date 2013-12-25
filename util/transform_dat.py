import pandas as pd
import numpy as np
from .parser import dat_parser

# The dimension of X should be 12,810, here we hardcode it in the function
N_COL = 12810 + 1

def fill_data(y, x_list, x_len=0):
    if x_len == 0:
        x_len = N_COL - 1
    filled_row = np.empty(x_len + 1)
    filled_row[:] = np.NAN
    filled_row[0] = y
    for x_loc, x_val in x_list:
        filled_row[int(x_loc)] = float(x_val)
    return filled_row

def make_df(DAT_PATH, n_col=0):
    if n_col == 0:
        n_col = N_COL
    df = pd.DataFrame(
        [fill_data(y, x_list) for y, x_list in dat_parser(DAT_PATH)],
        columns=[
            ("x_{}".format(i) if i >= 1 else "y") for i in range(N_COL)
        ]
    )
    return df

def make_csv(DAT_PATH, CSV_PATH, n_col=0):
    df_out = make_df(DAT_PATH, n_col)
    df_out.to_csv(CSV_PATH, index=False)

