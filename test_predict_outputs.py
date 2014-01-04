'''
Created on Jan 5, 2014

@author: c3h3
'''

from bows_celery.tasks import *

one_predict_job = ImageData.objects(class_id = 0)[0]

print "one_predict_job = ",one_predict_job
print one_predict_job.sift_desc_id



if __name__ == '__main__':
    pass