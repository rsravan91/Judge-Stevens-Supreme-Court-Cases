setwd("C:/Users/DELL/Desktop/ISSMTECHASSIGNMENTS/Kaggle/US Supreme Court/")
# Read the data for justice stevens
stevens=read.csv("stevens.csv")
str(stevens)
table(stevens$Reverse)

# split the datasets into train and test set

library("caTools")
set.seed(3000)
split=sample.split(Y = stevens$Reverse,SplitRatio = .70)
train=subset(stevens,split==TRUE)
test=subset(stevens,split==FALSE)

# Load the package for cart decsion tree i.e, rpart
library(rpart)
library(rpart.plot)

# Build the decision tree using method=class otherwise regression tree are built
# Provide minbucket to avoid overfitting, change this value to tune the tree
StevensTree=rpart(formula = Reverse~Circuit+Issue+Petitioner+Respondent+LowerCourt+Unconst,data = train,method="class",minbucket=25)
summary(StevensTree)
# plot the tree
prp(StevensTree)
# we see that variables are abbreviated in decision tree, we can see its expansion by creating table for all those variables
table(train$Respondent)

# We test the model on the test set
PredictCART=predict(StevensTree,test,type="class") # similar to threshold of 0.5 i.e, class would result in majority winner
table(test$Reverse,PredictCART)

# Build ROC curve
library(ROCR)
PredictROCR=predict(StevensTree,test)
# here we predict without class arguement as it provides a table with propotion of each target value in the subset it belongs
# here we use the second columns as the probabilit of 1 i.e, reversal happening
predictionROCR=prediction(PredictROCR[,2],labels = test$Reverse)
performanceROCR=performance(predictionROCR,"tpr","fpr")
plot(performanceROCR,colorize=TRUE,print.cutoffs.at=seq(0,1,0.1),text.adj=0)

# AUC
as.numeric(performance(predictionROCR,"auc")@"y.values")

# we can try diff values of minbucket and cutoffs/thresholds to further improve the cart model

# We build Random forest model for the above dataset
library(randomForest)
# We see that unlike CART we dont have method pararmeter to specify the model to build a classification rather than regression
# model, instead we feed the target variable as a factor variable to specify that RF should build a classification model
train$Reverse=as.factor(train$Reverse)
test$Reverse=as.factor(test$Reverse)
set.seed(200)
StevensForest=randomForest(Reverse~Circuit+Issue+Petitioner+Respondent+LowerCourt+Unconst,data=train,nodesize = 25,ntree = 200)
predictForest=predict(StevensForest,newdata = test)
table(test$Reverse,predictForest)
# Accuracy is 114/170 i.e ~67% which is better than CART model
# Random Forest accuracy would vary each time the forest is run as there is a random comoponent to the model as result of
# bagging which takes place when building the tree


# Build CART using Cross Validation using caret and e1071 package
library(caret)
library(e1071)
# Mention number of folds for the CV
# create 10 folds cross vaidation
numFolds=trainControl(method = "cv",number = 10)
# Pick possible values for cp parrameter to specify complexity
cpGrid=expand.grid(.cp=seq(0.01,0.5,0.01))
# Perform cross validation and build Random Forest
# trControl is the number and method of cross validation required, its numFolds here
# TuneGrid takes in possible values of cp
train(Reverse~Circuit+Issue+Petitioner+Respondent+LowerCourt+Unconst,data=train,method = "rpart",trControl = numFolds,tuneGrid = cpGrid)
# Output is table with accuracy with respect to the different cp parameters provided
# We see the accuracy increases as cp increases and then starts decreasing. Here accuracy was used to select the optimal model
# therefore we use this optimal cp value to build the CART model instead of the minbucket parameter
StevensTreeCV=rpart(formula = Reverse~Circuit+Issue+Petitioner+Respondent+LowerCourt+Unconst,data = train,method = "class",cp=0.19)
predictCV=predict(StevensTreeCV,newdata = test,type="class")
table(test$Reverse,predictCV)
prp(StevensTreeCV)
# Accuraccy of this model 0.724, which was better than previous cart model
# Thus by using CV we can select the best parameter value
# Thus similar to setevens remaining 8 judges were used to totally build 9 different models were used voting to decid the final judgement
# This approach was used to prdict 68 cases in Oct 2002 
  