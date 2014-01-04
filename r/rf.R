load("~/c3h3works/MLDM_Projects/2013-ML-TeamProject/dataset/training_data_zeros_60x60.RData")
load("~/c3h3works/MLDM_Projects/2013-ML-TeamProject/dataset/testing_data_zeros_60x60.RData")

library(fpc)

View(training_df[,-1])
rf_model <- randomForest(y ~ ., data=training_df,ntree=2000, mtry = 120, importance=TRUE,proximity=TRUE)

testing_df_pred = predict(rf_model,testing_df)
testing_df_pred2 = predict(rf_model,testing_df)
testing_df_pred3 = predict(rf_model,testing_df)
write.csv(testing_df_pred3,file="output_try_rf_nt2000_mt120.csv", quote = FALSE,row.names = FALSE)

save(rf_model, file="rf_model_nt2000_mt120.RData")

