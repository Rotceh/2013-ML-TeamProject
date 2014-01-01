load("~/c3h3works/MLDM_Projects/2013-ML-TeamProject/dataset/training_data_zeros_60x60.RData")
load("~/c3h3works/MLDM_Projects/2013-ML-TeamProject/dataset/testing_data_zeros_60x60.RData")


library(spls)

plsda_model = splsda(x=as.matrix(training_df[,-1]),y=as.matrix(training_df[,1]),K=40,eta=0.9,classifier="logistic",fit="kernelpls")

testing_df_pred = predict(plsda_model,testing_df)
testing_df_pred2 = predict(plsda_model,testing_df)
testing_df_pred3 = predict(plsda_model,testing_df)
write.csv(testing_df_pred3,file="output_Kplsda_K40_e0p9.csv", quote = FALSE,row.names = FALSE)

