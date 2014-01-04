'''
Created on Jan 4, 2014

@author: c3h3
'''

import pandas as pd
import cv2
from matplotlib import pylab as pl
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt


PIXEL_X = 60
PIXEL_Y = 60

def grey_convert(wd_array):
    """convert [0, 1] greyscale to [0, 255] (dark, white)"""
    return np.round((1 - wd_array) * 255).astype(np.uint8)


class ImageWords(list):
    __slots__ = ["ix", "zwd", "kp_list"]
    
    def __init__(self, ix, i_raw_df):
        self.ix = ix 
        self.zwd = i_raw_df[0]
        self.kp_list = []

    def update_keyword(self, kp, desc):
        self.append(desc)
        self.kp_list.append(kp)

    
    

    
sigma = 1.6 # default: 1.6
# If your image is captured with a weak camera with soft lenses,
# you might want to reduce the number.
contrastThreshold = 0.04 # default: 0.04
nOctaveLayers = 3  # default: 3
edgeThreshold = 5 # default: 10

sift = cv2.SIFT(
    sigma=sigma, 
    contrastThreshold=contrastThreshold,
    nOctaveLayers=nOctaveLayers,
    edgeThreshold=edgeThreshold
)



if __name__ == '__main__':
    pass

    # CSV_TRAIN = "../dataset/newtest_60x60_grey.csv" #"../dataset/train_zero_60x60.csv"
    
    csv_file_path = "../dataset/train_60x60_grey.csv"
    #csv_file_path = "../dataset/test_60x60_grey.csv"
    
    
    
    df_train = pd.read_csv(csv_file_path)
    print df_train.values.shape
    print df_train.values[:,0]
    
    one_img = df_train.values[0,1:]
    one_img1 = df_train.values[1,1:]
    print "one_img = ", one_img
    print "one_img[0] = ", one_img[0]
    
    kp, desc = sift.detectAndCompute(grey_convert(one_img.reshape(PIXEL_X, PIXEL_Y)),None)
    kp1, desc1 = sift.detectAndCompute(grey_convert(one_img1.reshape(PIXEL_X, PIXEL_Y)),None)
    
    print "kp = ", kp
    print len(kp)
    print dir(kp[0])
    #print kp[0].__dict__() 
    
    print "desc = ", desc
    
    #print dict(zip(kp,desc))
    
    
    norm = cv2.NORM_L2
    bfmatcher = cv2.BFMatcher(norm)
    

    # Match descriptors.
    print "type(desc) = ",type(desc)
    matches = bfmatcher.match(desc,desc1)
    
    print "matches = ",matches
    print "len(matches) = ",len(matches)
    
    one_matche_obj = matches[0]
    
    print map(lambda xx:xx.distance,matches)
    
#    img3 = cv2.drawMatches(one_img,kp, one_img1,kp, matches, flags=2)
#    plt.imshow(img3)
#    plt.show()

    
    
#    group_train = df_train.groupby("y")
#    print group_train
    

