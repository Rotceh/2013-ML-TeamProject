load("~/c3h3works/MLDM_Projects/2013-ML-TeamProject/dataset/training_data_zeros_60x60.RData")
load("~/c3h3works/MLDM_Projects/2013-ML-TeamProject/dataset/testing_data_zeros_60x60.RData")

library(fpc)

View(training_df[,-1])
dbscan_model = dbscan(data=training_df[1:2000,-1],eps=18,MinPts=5,method="hybrid")