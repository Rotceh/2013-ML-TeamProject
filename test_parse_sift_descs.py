'''
Created on Jan 5, 2014

@author: c3h3
'''

from bows_celery.tasks import *

csv_file_path = "dataset/train_60x60_grey.csv"
df_train = pd.read_csv(csv_file_path)
training_data = df_train.values

csv_file_path = "dataset/test_60x60_grey.csv"
df_test = pd.read_csv(csv_file_path)
testing_data = df_test.values


csv_file_path = "dataset/newtest_60x60_grey.csv"
df_new_test = pd.read_csv(csv_file_path)
new_testing_data = df_new_test.values

all_data = np.r_[training_data, testing_data, new_testing_data]

#all_data = testing_data
#print "all_data = ",all_data
#print "all_data.shape = ",all_data.shape

n_rows = all_data.shape[0]

print "n_rows = ",n_rows

for i in range(n_rows):
    print "i = ",i
    image_words = ImageWords(i,all_data[i,:])
    one_job = get_sift_descriptors.apply_async(kwargs = {"iws":image_words})
    image_data, created = ImageData.objects.get_or_create(ix = i)
    image_data.sift_desc_id = one_job.id 
    image_data.save()
    
    
    

