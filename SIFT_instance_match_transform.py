# coding: utf-8 -*-
from __future__ import division
import pandas as pd
import cv2
import numpy as np

from util.get_data import get_new_suite

SUITE_NAME = "60x60grey"
df_train, df_test, train_Y, train_X, test_Y, test_X = get_new_suite()

group_train = df_train.groupby("y")
group_inst_ix = [v for v in group_train.groups.values()]

test_ix_list = df_test.index
# part_group_inst_ix = [group_inst_ix[0], group_inst_ix[3]]


class Instance(object):
    __slots__ = ["ix", "zwd", "kp_list", "desc_list"]

    def __init__(self, ix, zwd, kp_list=None, desc_list=None):
        self.ix = ix
        self.zwd = zwd
        self.kp_list = [] if kp_list is None else kp_list
        self.desc_list = [] if desc_list is None else desc_list


zwd_inst_mat = [
    [Instance(ix=ix, zwd=zwd) for ix in ix_array]
    for ix_array, zwd in zip(group_inst_ix, range(1, 13))
]

zwd_test_inst_mat = [
    [Instance(ix=ix, zwd=0) for ix in test_ix_list]
]


PIXEL_X = 60
PIXEL_Y = 60

flatten_inst_arary = np.array(
    [inst for inst_array in zwd_inst_mat for inst in inst_array]
)

flatten_test_inst_arary = np.array(
    [inst for inst_array in zwd_test_inst_mat for inst in inst_array]
)


def grey_convert(wd_array):
    """convert [0, 1] greyscale to [0, 255] (dark, white)"""
    return np.round((1 - wd_array) * 255).astype(np.uint8)


sigma = 1.4  # default: 1.6
contrastThreshold = 0.03  # default: 0.04
nOctaveLayers = 5  # default: 3
edgeThreshold = 7  # default: 10
sift = cv2.SIFT(
    sigma=sigma,
    contrastThreshold=contrastThreshold,
    nOctaveLayers=nOctaveLayers,
    edgeThreshold=edgeThreshold
)

print("work on SIFT")

def detect_SIFT(inst):
    inst.kp_list, inst.desc_list = sift.detectAndCompute(
        grey_convert(train_X[inst.ix].reshape(PIXEL_X, PIXEL_Y)),
        None
    )

map(detect_SIFT, flatten_inst_arary)
map(detect_SIFT, flatten_test_inst_arary)

TOP_K = 20
N_ZODIAC = 12
out_dict = {'y': train_Y}
# Top K columns
top_column_names = [
    "z{}_top{}".format(zwd + 1, k + 1)
    for zwd in range(N_ZODIAC) for k in range(TOP_K)
]
# Mean columns
mean_column_names = [
    "z{}_mean".format(zwd + 1) for zwd in range(N_ZODIAC)
]
out_dict.update({
    col_name: 0 for col_name in top_column_names + mean_column_names
})
df_sift_train_out = pd.DataFrame(out_dict)
df_sift_train_out.columns = (
    ['y'] +
    top_column_names +
    mean_column_names
)

df_sift_test_out = df_sift_train_out.copy()
df_sift_test_out.y = 0

# FLANN Matcher
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=100)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)

DISTANCE_TH = 270

print("work on train")

for inst in flatten_inst_arary:
    for zwd, zwd_inst_list in zip(range(1, 13), zwd_inst_mat):
        prop_matched_list = []
        for zwd_inst in zwd_inst_list:
            # print(i)
            if zwd_inst.desc_list is None:
                # print("{} of zwd({}) has no features".format(zwd_inst.ix, zwd_inst.zwd))
                prop_matched_list.append(0)
                continue
            len_zwd_inst_features = len(zwd_inst.desc_list)
            all_matches = flann.knnMatch(inst.desc_list, zwd_inst.desc_list, k=len_zwd_inst_features)
            matches_mat = np.array([
                [m.distance for m in sorted(
                    [match for match in row], key=lambda x: x.trainIdx
                )] for row in all_matches]
            )

            prop_matched_list.append(
                np.sum(np.where(matches_mat < DISTANCE_TH, 1, 0).sum(axis=0) != 0) / len_zwd_inst_features
            )

        # write match result to data fame
        prop_matched_list.sort(reverse=True)
        df_sift_train_out.ix[inst.ix, "z{}_top1".format(zwd):"z{}_top{}".format(zwd, TOP_K)] = prop_matched_list[:TOP_K]
        df_sift_train_out.ix[inst.ix, "z{}_mean".format(zwd)] = np.mean(prop_matched_list)


print("work on test")

for inst in flatten_test_inst_arary:
    for zwd, zwd_inst_list in zip(range(1, 13), zwd_inst_mat):
        prop_matched_list = []
        for zwd_inst in zwd_inst_list:
            # print(i)
            if zwd_inst.desc_list is None:
                # print("{} of zwd({}) has no features".format(zwd_inst.ix, zwd_inst.zwd))
                prop_matched_list.append(0)
                continue
            len_zwd_inst_features = len(zwd_inst.desc_list)
            all_matches = flann.knnMatch(inst.desc_list, zwd_inst.desc_list, k=len_zwd_inst_features)
            matches_mat = np.array([
                [m.distance for m in sorted(
                    [match for match in row], key=lambda x: x.trainIdx
                )] for row in all_matches]
            )

            prop_matched_list.append(
                np.sum(np.where(matches_mat < DISTANCE_TH, 1, 0).sum(axis=0) != 0) / len_zwd_inst_features
            )

        # write match result to data fame
        prop_matched_list.sort(reverse=True)
        df_sift_test_out.ix[inst.ix, "z{}_top1".format(zwd):"z{}_top{}".format(zwd, TOP_K)] = prop_matched_list[:TOP_K]
        df_sift_test_out.ix[inst.ix, "z{}_mean".format(zwd)] = np.mean(prop_matched_list)


CSV_SIFT_TRANSFORM_TRAIN = "dataset/train_60x60_grey_SIFT.csv"
CSV_SIFT_TRANSFORM_NEWTEST = "dataset/newtest_60x60_grey_SIFT.csv"
df_sift_train_out.to_csv(CSV_SIFT_TRANSFORM_TRAIN, index=False)
df_sift_test_out.to_csv(CSV_SIFT_TRANSFORM_NEWTEST, index=False)
