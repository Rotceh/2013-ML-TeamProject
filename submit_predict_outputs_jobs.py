'''
Created on Jan 5, 2014

@author: c3h3
'''

from bows_celery.tasks import *
import random



#one_predict_job = ImageData.objects(class_id = 0)[1]
#one_predict_job = ImageData.objects(ix = 1)[0]

for one_predict_job in ImageData.objects(class_id = 0):

    print "one_predict_job = ",one_predict_job
    one_ix = one_predict_job.ix
    
    one_ix_data = ImageData.objects(ix = one_ix)[0]
    print "one_ix_data = ",one_ix_data
    
#    print "get_sift_descs(one_ix_data) = ",get_sift_descs(one_ix_data)
#    print "get_sift_descs(one_ix_data).info = ",get_sift_descs(one_ix_data).info
    
    all_classes = range(1,13)
    print all_classes
    
    for one_class in all_classes:
    
        one_class_all_data = ImageData.objects(class_id = one_class)
        
        print " len(one_class_all_data) = ",len(one_class_all_data)
        
        sample_data = random.sample(one_class_all_data, 200)
        
        print "sample_data = ",sample_data
            
            #one_class_all_data = ImageData.objects(ix = 22)
            
        #for xx in one_class_all_data:
        for xx in sample_data:
            
                
            
#            print "xx = ",xx 
#            print "get_sift_descs(xx).info = ",get_sift_descs(xx).info
#            print "len(get_sift_descs(xx).info) = ",len(get_sift_descs(xx).info)
#            print "len(get_sift_descs(one_ix_data).info) = ",len(get_sift_descs(one_ix_data).info)
            
            print compute_two_image_matching_scores.apply_async(kwargs = {"img1_data":one_ix_data, "img2_data":xx})
        
#        norm = cv2.NORM_L2
#        bfmatcher = cv2.BFMatcher(norm)
#        
#        MATCH_DIST_UPPER_BOUND = 300    
#        
#        
#        # Match descriptors.
#        matches = bfmatcher.match(np.array(get_sift_descs(xx).info),
#                                  np.array(get_sift_descs(one_ix_data).info))
#        
#        print matches
#        print "len(matches) = ",len(matches)
#        filtered_matches = [x for x in matches if x.distance <= MATCH_DIST_UPPER_BOUND]
#        print "len(filtered_matches) = ",len(filtered_matches)
#        
#        
#        matching_score = 0 if len(matches) == 0 else float(len(filtered_matches)) / float(len(matches))
#        
#        print "matching_score = ",matching_score
#        
#        
#        score_data, created = NewMatchingScores.objects.get_or_create(ix1 = one_ix_data.ix, 
#                                                                      c1 = one_ix_data.class_id,
#                                                                      ix2 = xx.ix, 
#                                                                      c2 = xx.class_id)
#        score_data.score = matching_score
#        score_data.save()
#    
#        print "score_data = ",score_data


#map(lambda ImageData.objects(class_id = one_class))


if __name__ == '__main__':
    pass
