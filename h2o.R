###################
# Title : Wisconsin Breast Cancer data set
# Data Source: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original) 
# Author : Yesenia Silva
# MSDS664x70_Stat Infer Predictive Analytics
###################

##############Packages#############
library(readr)  ##Read the file
library(h2o)
library(mlbench) ##Get datafile

##Load File to R
BreastCancer <- read_csv("C:/Users/ysilva/Desktop/Statistical Inference and Predictive Analytics/breastcancer.csv")
str(BreastCancer) #bareneclei is coming in as character but shoudl be numeric
BreastCancer$barenuclei #noticed many ? characters
is.na(BreastCancer) <- BreastCancer == "?" #Change ? no NA
summary(BreastCancer)
BreastCancer$barenuclei <- as.numeric(BreastCancer$barenuclei) #Change to numeric and see 12 NAs

#Remove NAs 
na.omit(BreastCancer)
## turn class to factor
BreastCancer$class <- as.factor(BreastCancer$class)


## Start a local cluster with 1GB RAM (default)
localH2O <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE)

##Load data into h2oframe obje3ct
summary(BreastCancer)
dat <- BreastCancer[, -1] # remove the ID column
class(dat)
dat[, c(1:ncol(dat))] <- sapply(dat[, c(1:ncol(dat))], as.numeric) # Convert factors into numeric
## Convert Breast Cancer into H2O
dat.h2o <- as.h2o(dat, destination_frame = "midata")

# #path to file
# getwd()
# path <- "C:/Users/ysilva/Documents/breastcancer.csv"
#   
# 
# ##link the datasets to the H2O cluster
# dat <- h2o.uploadFile(path = path)

#Verify h2o frame 
class(dat.h2o)
str(dat.h2o)

#Look a response output
h2o.table(dat.h2o$class)

#--------------Partition the data into train and test sets------------#
#Create a vector of random and uniform numbers for the full data
rand <- h2o.runif(dat.h2o, seed = 123)
#build your partitioned data and assign it with a desired key name
train <- dat.h2o[rand <= 0.7, ]
train <- h2o.assign(train, key = "train")
test <- dat.h2o[rand > 0.7, ]
test <- h2o.assign(test, key = "test")
#use h2o.table() to check that we have a balanced response variable between the train and test sets
h2o.table(train[, 10])
h2o.table(test[, 10])
#---------------Training a Deep Neural Network Model-----------#
model2 <- 
  h2o.deeplearning(x = 2:785, 
                   y = 1,   
                   training_frame = train, # train data in H2O format
                   activation = "TanhWithDropout", # or 'Tanh'
                   input_dropout_ratio = 0.2, # % of inputs dropout
                   hidden_dropout_ratios = c(0.5,0.5,0.5), # % for nodes dropout
                   balance_classes = FALSE, 
                   hidden = c(50,50,50), # three hidden with 50 nodes each
                   epochs = 100, # max. no.# of passes over the training data
                   nfolds = 5, #  N-fold cross-validation will be performed
                   variable_importances=T) # not enabled by default
#Model w/ 3 hidden layers of 20 nodes each.
model3 <- 
  h2o.deeplearning(x = 2:785, 
                   y = 1,   
                   training_frame = train, # train data in H2O format
                   activation = "TanhWithDropout", # or 'Tanh'
                   input_dropout_ratio = 0.2, # % of inputs dropout
                   hidden_dropout_ratios = c(0.5,0.5,0.5), # % for nodes dropout
                   balance_classes = FALSE, 
                   hidden = c(20,20,20), # three hidden with 50 nodes each
                   epochs = 100, # max. no.# of passes over the training data
                   nfolds = 5, #  N-fold cross-validation will be performed
                   variable_importances=T) # not enabled by default
#Model w/ 2 hidden layers of 30 nodes 
model4 <- 
  h2o.deeplearning(x = 2:785, 
                   y = 1,   
                   training_frame = train, # train data in H2O format
                   activation = "TanhWithDropout", # or 'Tanh'
                   input_dropout_ratio = 0.2, # % of inputs dropout
                   hidden_dropout_ratios = c(0.5,0.5), # % for nodes dropout
                   balance_classes = FALSE, 
                   hidden = c(30,30), # two hidden with 30 nodes each
                   epochs = 100, # max. no.# of passes over the training data
                   nfolds = 5, #  N-fold cross-validation will be performed
                   variable_importances=T) # not enabled by default
#examine the performance on the holdout folds
model1
print(model1)
summary(model2)
plot(model1)
#----------------------Using the Model for Prediction---------#
## Using the DNN model for predictions
h2o_yhat_test <- h2o.predict(model1, test)

## Converting H2O format into data frame
df_yhat_test <- as.data.frame(h2o_yhat_test)
df_yhat_test

#examine performance
perf <- h2o.performance(model3, test)
perf

#confusion matrix
h2o.confusionMatrix(model3)
