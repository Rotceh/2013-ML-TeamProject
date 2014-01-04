'''
Created on Jan 5, 2014

@author: c3h3
'''


from __future__ import absolute_import

from bows_celery.celery import celery

import cv2
import pandas as pd
import cv2
from matplotlib import pylab as pl
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt


import mongoengine as models

models.connect("image_data_db")


class ImageData(models.Document):
    ix = models.IntField()
    class_id = models.IntField()
    sift_desc_id = models.StringField()
    parsing_status = models.StringField(default = "waiting")
    
    def __unicode__(self):
        return "[image %s / class %s / %s / %s]" % (self.ix, self.class_id, self.sift_desc_id, self.parsing_status)





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


def grey_convert(wd_array):
    """convert [0, 1] greyscale to [0, 255] (dark, white)"""
    return np.round((1 - wd_array) * 255).astype(np.uint8)

 
PIXEL_X = 60
PIXEL_Y = 60
 
class ImageWords(list):
    __slots__ = ["ix", "zwd", "_image", "kp_list"]

    def __unicode__(self):
        return "[image %s / class %s]" % (self.ix, self.zwd)
    

    def __init__(self, ix, i_raw_df):
        self.ix = ix 
        self.zwd = i_raw_df[0]
        self._image = i_raw_df[1:]
        self.kp_list = []

        self._image = grey_convert(self._image.reshape(PIXEL_X, PIXEL_Y))

    def update_keyword(self, kp, desc):
        self.append(desc)
        self.kp_list.append(kp)

    def get_kps_descs(self):
        
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
        
        kps, descs = sift.detectAndCompute(self._image,None)
        
        if descs != None:
            self.extend(descs)
        if kps != None:
            self.kp_list.extend(map(lambda one_kp:{'angle':one_kp.angle,
                                                   "class_id":int(self.zwd), 
                                                   'pt':one_kp.pt, 
                                                   'size':one_kp.size},kps))
        
        


@celery.task(ignore_result = True)
def finished_parsing_descriptors(iws):
    image_data, created = ImageData.objects.get_or_create(ix = iws.ix)
    image_data.parsing_status = "finished"
    image_data.class_id = int(iws.zwd)
    image_data.save()



@celery.task
def get_sift_descriptors(iws):
    print "iws = ",iws
    iws.get_kps_descs()
    finished_parsing_descriptors.apply_async(kwargs = {"iws":iws})
    return iws 
    
    







if __name__ == '__main__':
    
    pass