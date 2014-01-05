load("~/c3h3works/MLDM_Projects/2013-ML-TeamProject/dataset/training_data_zeros_60x60.RData")
load("~/c3h3works/MLDM_Projects/2013-ML-TeamProject/dataset/testing_data_zeros_60x60.RData")

library(fpc)

View(training_df[,-1])
rf_model <- randomForest(y ~ ., data=training_df,ntree=2000, mtry = 30, importance=TRUE,proximity=TRUE)

testing_df_pred = predict(rf_model,testing_df)
sign_newtest_60x60_grey_pred = predict(rf_model,sign_newtest_60x60_grey[,-1])
write.csv(sign_newtest_60x60_grey_pred,file="output_newest_rf.csv", quote = FALSE,row.names = FALSE)

save(rf_model, file="rf_model_default.RData")