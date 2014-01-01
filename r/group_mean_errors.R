
group_mean <- ddply(train_zero_60x60, .(y), mean)

group_mean_without_labels <- group_mean[,-1]
train_zero_60x60_without_labels = train_zero_60x60[,-1]

one_row = train_zero_60x60_without_labels[1,]
compute_groups_errors <- function(one_row){
  return(apply(group_mean_without_labels,1, function(xx) return( mean((xx - one_row)^2) )))
}

new_features = apply(train_zero_60x60_without_labels,1,compute_groups_errors)

new_features_df = data.frame(y=as.factor(train_zero_60x60[,1]),t(new_features))
View(new_features_df)
save(new_features_df,file="new_features_groups_errors.RData")


new_testing_features = apply(testing_df,1,compute_groups_errors)
new_testing_features_df = data.frame(t(new_testing_features))
save(new_testing_features_df,file="new_features_groups_errors_testing.RData")


library(e1071)
svm_models <- svm(y~.,data=new_features_df, C = 100, gamma = 0.01)
pred_new_testing_features_df <- predict(svm_models, new_testing_features_df)

