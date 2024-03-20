df <- read.csv("creditworthiness.csv")
#remove credit rating = 0
df_clean<- df[!(df$credit.rating == 0),]
View(df_clean)
unique(df_clean$credit.rating)

#change data type of credit rating
df_clean$credit.rating <- as.factor(df_clean$credit.rating)

#train test split
set.seed(123)  # Set a random seed for reproducibility
train_idx <- sample(1:nrow(df_clean), 0.5 * nrow(df_clean))  # 50% for training
train_data <- df_clean[train_idx, ]
test_data <- df_clean[-train_idx, ]

#train model
tree_model <- rpart(credit.rating ~ ., data = train_data)
tree_model
summary(tree_model)
#plot model
rpart.plot(tree_model)
plot(tree_model)
text(tree_model))

#predict model
predictions <- predict(tree_model, newdata = test_data, type = "class")

actual_classes <- test_data$credit.rating
# Calculate the confusion matrix
confusion <- table(predictions, actual_classes)

# Calculate accuracy
accuracy <- sum(diag(confusion)) / sum(confusion)
accuracy
# Calculate precision
precision <- diag(confusion) / colSums(confusion)

# Calculate recall (Sensitivity)
recall <- diag(confusion) / rowSums(confusion)

# Calculate F1 score
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print evaluation metrics
print(confusion)
print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1 Score:", f1_score))
# Plot the confusion matrix as a heatmap
library(ggplot2)

# Plot the confusion matrix as a heatmap
heatmap(confusion, col = heat.colors(length(unique(actual_classes))), 
        main = "Confusion Matrix", xlab = "Predicted", ylab = "Actual")




#2e
#Random Forest
# Load the ranger library
library(ranger)

# Specify the formula for the model
formula <- credit.rating ~ .

# Create the random forest model
rf_model <- ranger(formula, data = train_data)

# Print the model output
print(rf_model)

median_customer <- c(0,1,1,0,3,0,3,3,0,3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3)
length(median_customer)


# Split the train and test datasets into predictors and target variables
train_predictors <- train_data[, -which(names(train_data) == "credit.rating")]
train_target <- train_data$credit.rating

test_predictors <- test_data[, -which(names(test_data) == "credit.rating")]
test_target <- test_data$credit.rating
# Fit the SVM model using the default settings
svm_model <- svm(train_predictors, train_target)


# Make predictions on the test data
svm_predictions <- predict(svm_model, newdata = test_predictors)

# Print the predictions
print(svm_predictions)


# Calculate the accuracy
accuracy <- sum(svm_predictions == test_target) / length(test_target)
print(accuracy)


confusion_matrix <- table(svm_predictions, test_target)
print(confusion_matrix)

parameter_grid <- expand.grid(
  C = c(0.01, 0.1, 1),            # Penalty parameter C
  gamma = c(0.1, 1, 10),           # Kernel parameter gamma (only for rbf kernel)
  kernel = c("linear", "rbf")     # Kernel type
)
#svm_tuned <- tune(svm, credit.rating ~ ., data = train_data, ranges = parameter_grid)

# Create the Naive Bayes model
nb_model <- naiveBayes(credit.rating ~ ., data = train_data)

# Make predictions on the test data
nb_predictions <- predict(nb_model, newdata = test_data)
length(nb_predictions)
length(test_data$credit.rating)
# Evaluate the model performance
nb_confusion_matrix <- table(nb_predictions, test_data$credit.rating)
accuracy <- sum(diag(nb_confusion_matrix)) / sum(nb_confusion_matrix)

print(nb_confusion_matrix)
print(accuracy)

#get data of median customer
median_customer <- list(0,1,1,0,3,0,3,3,0,3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3)
name_vector = as.vector(colnames(df_clean))
new_name_vector = name_vector[-length(name_vector)]
df_median_customer <- data.frame(setNames(median_customer, new_name_vector))
df_median_customer

nb_predict_median_customer <- predict(nb_model, newdata = df_median_customer,type = "raw")
nb_predict_median_customer

summary(nb_model)

library (dplyr)
df_binary <-df_clean
df_binary <- dplyr::mutate(df_binary,credit_rating_binary = ifelse(credit.rating == 1, 1, 0))
df_binary

#drop the old column credit rating
df_binary <- select(df_binary, -credit.rating)
df_binary

# Split your data into training and testing sets
set.seed(123)
train_indices <- sample(1:nrow(df_binary), 0.5 * nrow(df_binary))
train_data_binary <- df_binary[train_indices, ]
test_data_binary <- df_binary[-train_indices, ]
test_data_binary
# Create the logistic regression model
logreg_model <- glm(credit_rating_binary ~ ., data = train_data_binary, family = binomial)
summary(logreg_model)
# Make predictions on the test data
predicted_probs <- predict(logreg_model, newdata = test_data_binary, type = "response")

# Convert predicted probabilities to predicted class labels
predicted_labels <- ifelse(predicted_probs > 0.5, 1, 0)
predicted_labels
# Evaluate the model performance (e.g., accuracy, confusion matrix, etc.)
accuracy <- sum(predicted_labels == test_data_binary$credit_rating_binary) / nrow(test_data_binary)

print(accuracy)

confusion_log <- table(predicted_labels, test_data_binary$credit_rating_binary)
confusion_log

svm_model <- svm(credit_rating_binary ~ ., data = train_data_binary, kernel = "radial")

# Print the model summary
print(svm_model)

decisiontree_predict_median_customer <- predict(tree_model, newdata = df_median_customer,type = "prob")
decisiontree_predict_median_customer
summary(decisiontree_predict_median_customer)





split_variable <- tree_model$frame$var[1]  # Variable used for the split
left_child_counts <- tree_model$frame$nc[1, ]  # Class counts in the left child node
right_child_counts <- tree_model$frame$nc[2, ]  # Class counts in the right child node

# Define function to calculate entropy
entropy <- function(p) {
  -sum(p * log2(p))
}

# Calculate entropy before the split
entropy_S <- entropy(tree_model$frame$prob[1, ])

# Calculate entropy in left child node
entropy_left <- entropy(left_child_counts / sum(left_child_counts))
# Calculate entropy in right child node
entropy_right <- entropy(right_child_counts / sum(right_child_counts))

# Calculate weighted average of entropy in left and right child nodes
weighted_entropy <- (sum(left_child_counts) / sum(tree_model$frame$nc)) * entropy_left +
  (sum(right_child_counts) / sum(tree_model$frame$nc)) * entropy_right

# Calculate gain in entropy
gain_entropy <- entropy_S - weighted_entropy

# Print the gain in entropy
cat("Gain in entropy:", gain_entropy)