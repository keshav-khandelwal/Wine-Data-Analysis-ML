# Install and load the necessary package
# install.packages("rattle")
library(rattle)
# rattle is used to load wine dataset
# Convert the wine dataset into a dataframe
data<-as.data.frame(wine)
View(data)
# Check the structure of the dataset
str(data)
# Get summary statistics for the dataset
summary(data)
# Check for missing values
# Count total missing values in the dataset
# Count missing values in each column
sum(is.na(data))
colSums(is.na(data))
missingdata<-data[!complete.cases(data), ]
sum(is.na(missingdata))
View(data)
# Split the dataset into training and testing sets
# Load the caTools library for splitting data
library(caTools)
set.seed(123)
# Split 70% data for training and 30% for testing
x = sample.split(data$Type,SplitRatio = 0.70)
View(x)
# Extract training set
train = subset(data,x==TRUE)
# Extract testing set
test = subset(data,x==FALSE)
# Separate features and labels for training and testing
train_label = train[,1:14]hashtag#features in training
test_label =test[,1:14]hashtag#features in testing
train_n = train[,1]hashtag#target variable
test_n = test[,1]#
# KNN Classification
# Load the class library for KNN
# Perform KNN classification with k = 5
library(class)
pred = knn(train_label,test_label,train_n,k=5)
pred
hashtag#confusion matrix
library(caret)
cm = table(pred,test_n)
cm
hashtag#precision,recall and f1 score for knn
precision<-posPredValue(cm,positive = 1)
precision
recall <- sensitivity(cm,positive = 1)
recall
KNN_f1 <- (2*precision*recall)/(precision+recall)
KNN_f1
hashtag#Calculate accuracy and errorate for knn
accuracy = sum(diag(cm)) / sum(cm)
accuracy
errorrate = 1-accuracy
errorrate
library(gmodels)
CrossTable(x=test_n,y=pred,prop.chisq=FALSE)
hashtag#Load the libraries for naive bayes classification
library(e1071)
library(caret)
library(caTools)
hashtag#NaiveBayes Classificatio
classifier <- naiveBayes(Type~.,data=train)
classifier
y_pred<-predict(classifier,newdata=test)
hashtag#Confusion Matrix
cm<-table(test$Type,y_pred)
cm
hashtag#Precision ,Recall,f1-score
precision<-posPredValue(cm,positive = 1)
precision
recall <- sensitivity(cm,positive = 1)
recall
naive_f1 <- (2*precision*recall)/(precision+recall)
naive_f1
hashtag#Calculating Accuracy and error rate
naive_accu<-sum(diag(cm))/sum(cm)
naive_accu
naive_error <- 1- naive_accu
naive_error
hashtag#load libraries for Decision tree
library(rpart)
hashtag#train model for decision tree
model<-rpart(Type~.,data=train,method="class")
plot(model)
text(model)
library(rpart.plot)
rpart.plot(model)
rpart.plot(model,type=4,extra=103)
hashtag#prediction using decision tree model
pred = predict(model,test,type="class")
cm<-table(pred,test$Type)
cm
hashtag#precision,recall,f1-scores for DEcision tree
precision<-posPredValue(cm,positive = 1)
precision
recall <- sensitivity(cm,positive=1)
recall
decision_f1 <- (2*precision*recall)/(precision+recall)
decision_f1
hashtag#Accuracy and Error rate for Decision tree
decision_accu <-sum(diag(cm))/sum(cm)
decision_accu
decision_error<-1-decision_accu
decision_error
hashtag#Multiple Linear regression
regressor = lm(formula = Alcohol ~ Malic + Ash + Alcalinity + Magnesium + Phenols + Flavanoids + Nonflavanoids + Proanthocyanins + Color + Hue + Dilution + Proline, data = train)
# Make predictions on the test set
y_pred = predict(regressor, newdata = test)
# Display predicted values
y_pred
CrossTable(x=test_n,y=y_pred,prop.chisq=FALSE)
# Check the correlation matrix of the training set features
a = cor(train[, c("Malic", "Ash", "Alcalinity", "Magnesium", "Phenols", "Flavanoids", "Nonflavanoids", "Proanthocyanins", "Color", "Hue", "Dilution", "Proline")])
# Visualize the correlation matrix using 'corrplot'
library(corrplot)
corrplot(a, method = "number")
corrplot(a, type = "upper")
# Visualize relationships between variables using pairs
library(psych)
pairs(train[, c("Malic", "Ash", "Alcalinity", "Magnesium", "Phenols", "Flavanoids", "Nonflavanoids", "Proanthocyanins", "Color", "Hue", "Dilution", "Proline")])
pairs.panels(train[, c("Malic", "Ash", "Alcalinity", "Magnesium", "Phenols", "Flavanoids", "Nonflavanoids", "Proanthocyanins", "Color", "Hue", "Dilution", "Proline")])
# Summary of the regression model
summary(regressor)
# Visualizing predictions vs Actual values using ggplot2
library(ggplot2)
ggplot(data = test, aes(x = Alcohol, y = y_pred)) +
 geom_point(color = "blue") +
 geom_smooth(method = "lm", color = "red") +
 labs(title = "Predicted vs Actual Alcohol Content", x = "Actual Alcohol", y = "Predicted Alcohol")
hashtag#neural networks
library(neuralnet)
# Define formula for neural network
nn_formula <- Type ~ Malic + Ash + Alcalinity + Magnesium + Phenols + Flavanoids + Nonflavanoids + Proanthocyanins + Color + Hue + Dilution + Proline
# Train the neural network model
nn_model <- neuralnet(nn_formula, data = train, hidden = c(5), linear.output = FALSE)
# Plot the model
plot(nn_model)
# Predict on test data
test_predictions <- compute(nn_model, test[, -1])$net.result
# Convert predictions to class labels
predicted_class <- apply(test_predictions, 1, which.max)
# Create the confusion matrix
cm<- table(Predicted = predicted_class, Actual = test$Type)
cm
# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
hashtag#precision,recall and f1 score for knn
precision<-posPredValue(cm,positive = 1)
precision
recall <- sensitivity(cm,positive = 1)
recall
neural_f1 <- (2*precision*recall)/(precision+recall)
neural_f1
hashtag#Calculate accuracy and errorate for knn
accuracy
errorrate = 1-accuracy
errorrate
