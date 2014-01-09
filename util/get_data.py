import pandas as pd

CSV_RAW_TRAIN = "dataset/train_na2zero.csv"
CSV_RAW_TEST = "dataset/test_na2zero.csv"
CSV_60x60_TRAIN = "dataset/train_zero_60x60.csv"
CSV_60x60_TEST = "dataset/test_zero_60x60.csv"
CSV_60x60addmean_TRAIN = "dataset/train_60x60_bw_addmean.csv"
CSV_60x60grey_TRAIN = "dataset/train_60x60_grey.csv"
CSV_60x60grey_TEST = "dataset/test_60x60_grey.csv"

def get_train_test_suite(suite_name="raw"):
    """Return df_train, df_test, train_Y, train_X, test_Y, test_X in order.

    Create full train/test dataframe and X, Y matrix.

    Parameters
    ==========
    suite_name : "raw", suite
        expect "raw", "60x60", "60x60grey", "60x60addmean"
    """
    if suite_name.lower() == "raw":
        CSV_TRAIN = CSV_RAW_TRAIN
        CSV_TEST = CSV_RAW_TEST
    elif suite_name.lower() == "60x60":
        CSV_TRAIN = CSV_60x60_TRAIN
        CSV_TEST = CSV_60x60_TEST
    elif suite_name.lower() == "60x60grey":
        CSV_TRAIN = CSV_60x60grey_TRAIN
        CSV_TEST = CSV_60x60grey_TEST
    elif suite_name.lower() == "60x60addmean":
        CSV_TRAIN = CSV_60x60addmean_TRAIN
        CSV_TEST = CSV_60x60_TEST
    else:
        raise ValueError(
            "Unexpected suite_name: {}, cannot handle.".format(suite_name)
        )

    df_train = pd.read_csv(CSV_TRAIN)
    df_test = pd.read_csv(CSV_TEST)

    train_Y = df_train.y
    train_X = df_train.iloc[:, 1:].values

    test_Y = df_test.y
    test_X = df_test.iloc[:, 1:].values

    return df_train, df_test, train_Y, train_X, test_Y, test_X


def get_new_suite(suite_name="60x60grey"):

    """Return df_train, df_test, train_Y, train_X, test_Y, test_X in order.

    Create full train/test dataframe and X, Y matrix.
    Train dataframe combined train and first half test data(with label).
    Test dataframe will be the second half test data.

    Parameters
    ==========
    suite_name : "60x60grey", str
        expect "raw", "60x60grey"
    """

    norm_suite_name = suite_name.lower()

    if norm_suite_name == "raw":
        CSV_TRAIN = "dataset/train_na2zero.csv"
        CSV_LABEL_TEST = "dataset/test_na2zero.csv"
        CSV_NEW_TEST = "dataset/newtest_na2zero.csv"
    elif norm_suite_name == "60x60grey":
        CSV_TRAIN = "dataset/train_60x60_grey.csv"
        CSV_LABEL_TEST = "dataset/test_60x60_grey.csv"
        CSV_NEW_TEST = "dataset/newtest_60x60_grey.csv"
    else:
        raise ValueError(
            "Unexpected suite_name: {}, cannot handle.".format(suite_name)
        )

    df_train = pd.concat([
        pd.read_csv(CSV_TRAIN),
        pd.read_csv(CSV_LABEL_TEST)
    ])
    df_test = pd.read_csv(CSV_NEW_TEST)

    train_Y = df_train.y
    train_X = df_train.iloc[:, 1:].values

    test_Y = df_test.y
    test_X = df_test.iloc[:, 1:].values

    return df_train, df_test, train_Y, train_X, test_Y, test_X

