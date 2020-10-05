############################ R Project semi-conductor #############################
# ## Author's information 
# Name : AeLee Im
# The Best approaches to find out the best Model
###################################################################################

################################ 0. System setting ###################################
#------------------------------------------------------------------------------------------------------
# Determine the folder of the current script file - works only in RStudio
#------------------------------------------------------------------------------------------------------
# OUTPUT:	  folder
#------------------------------------------------------------------------------------------------------
script.folder<-function()
{
  # load library
  p_load(rstudioapi)
  
  result <- dirname(rstudioapi::getActiveDocumentContext()$path)  
  
  return(result)
}

# set working directory
setwd(script.folder())

# === PREPARE SYSTEM
# set language to English
Sys.setenv(LANG = "en")

# In library pacman the function p_load can be used to load and install packages automatically
# for details see also 
# https://cran.r-project.org/web/packages/pacman/vignettes/Introduction_to_pacman.html
# https://darksky.net/poweredby/
#
# Check if pacman package itself is being installed, otherwise install it
if(!require("pacman")) install.packages("pacman")

# Step 1: Install and load R Packages 
library(pacman)

################################# 0-1. Data pre-processing ###################################

##### Load input data (SECOM dataset) #####

# p_load from pacman checks if the package is installed and installs as well as loads in library
# here we need data.table package for some of the functions
p_load(data.table)
p_load(curl)

# Load data file from the URL provided
secom_data_001 <- fread("https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data")

# converting the dowloaded dataset into a dataframe
secom_data_001 <- as.data.frame(secom_data_001)  
# to view loaded data click on the datarame displayed in variable viewer

# dowloading labels dataset
secom_labels <- fread("https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data")
secom_labels <- as.data.frame(secom_labels)

##### Adjust dataset(Valuable's name & feature combination etc.) #####

# Renaming All Columns as feature 1 to feature n
colnames(secom_data_001) <- paste0("Feature", 1: length(secom_data_001))

# Renaming All rows as semiconductor 1 to n
row.names(secom_data_001) <- paste0("SC_", 1: nrow(secom_data_001))

# Renaming columns for labels
colnames(secom_labels) <- c('Status','Timestamp')

############################ 0-2. Data pre-processing and Missing value analysis ###############################

########## Data cleasing 

#Graph showing missing values percent per column before % missing values removal

# Determine the missing value percentage
x <- as.data.frame(colMeans(is.na(secom_data_001)) * 100)
x$index <- c(1: nrow(x))
colnames(x) <- c("percentage","index")
#View(x)
plot(x$index, x$percentage, main = "Percent Missing Values Per Feature",
     xlab = "Feature Number", ylab = "Missing percent",
     pch = 20, frame = TRUE, col = "blue")

##### Remove columns with missing values more than 55%, and creating new dataset #####
secom_data_002 <- secom_data_001[, -which(colMeans(is.na(secom_data_001)) > 0.55)]
# -> 24 columns are removed

# Graph showing missing values percent per column After % missing values removal 
x2 <- as.data.frame(colMeans(is.na(secom_data_002))*100)
x2$index <- c(1:nrow(x2))
colnames(x2) <- c("percentage","index")
plot(x2$index, x2$percentage, main = "Percent Missing Values Per Feature",
     xlab = "Feature Number", ylab = "Missing percent", ylim = c(0,100),
     pch = 20, frame = TRUE, col = "blue")

##### Remove columns with constant values 
p_load(caret)
# Logic here is any column with constant values will have Zero Variance, delete that column
# for whole set
secom_data_003 <- secom_data_002[, -nearZeroVar(secom_data_002)]
# -> 127 more features were removed

##### Scaling / Normalising values 
p_load(ggplot2)
p_load(reshape2)

# just for visulization! this graph shows distribution of the data before scaling
secom_tri1 <- secom_data_003
secom_tri1$time <- 1:nrow(secom_data_003)
df3 <- melt(secom_tri1, id.vars = 'time', variable.name = "series")

ggplot(df3, aes(time, value)) + geom_line()


### Scaling: Here we rescale values between -1 and 1

p_load(BBmisc)
# scaling for training set
secom_data_004 <- BBmisc::normalize(secom_data_003, method = "range",range = c(-1, 1), margin = 1L, on.constant = "quiet")

# just for visulization! this graph shows distribution of the data after scaling
secom_tri2 <- secom_data_004
secom_tri2$time <- 1:nrow(secom_data_004)
df2 <- melt(secom_tri2, id.vars = 'time', variable.name = "series")

ggplot(df2, aes(time, value)) + geom_line() + ylim(-2, 2)

# complete the dataset with predictor data 
secom_data_004_bind <- cbind(secom_data_004, secom_labels)
# eliminate "Timestamp" column, which is not required for our further analysis
secom_data_004_bind <- secom_data_004_bind[c(-ncol(secom_data_004_bind))] 
head(secom_data_004_bind)
dim(secom_data_004_bind)

############################ 1. Split the dataset into train and test datasets ##################
# train : test dataset (80:20)

p_load(caTools)
set.seed(500)

split = sample.split(secom_data_004_bind$Status, SplitRatio =0.8)
training_set = subset(secom_data_004_bind, split == TRUE)
test_set_knn = subset(secom_data_004_bind, split == FALSE)

############################## 1. Missing value imputation #######################################

# checking for number of missing values
# if there is missing value, we have to impute the missing values
sum(is.na(secom_data_004_bind)) 
# -> As a result, we found out 14006 missing values in out dataset 

##### Missing value imputation 1. KNN - Impute missing values using KNN from follwing library #####

# load required library
p_load(DMwR)

# knn imputation for whole data 
secom_data_knn_all <-  knnImputation(secom_data_004_bind, k = 10, scale = F, meth = "weighAvg", 
                                 distData = NULL)
secom_data_knn_all

# knn imputation for train data 
secom_data_knn_all <-  knnImputation(training_set, k = 10, scale = F, meth = "weighAvg", 
                                     distData = NULL)
secom_data_knn_all

# knn imputation for test data 
secom_data_knn_test <-  knnImputation(test_set_knn, k = 10, scale = F, meth = "weighAvg", 
                                     distData = NULL)
secom_data_knn_test

# checking if there are missing values after knn imputation
sum(is.na(secom_data_knn_all))
# -> As a result, every missing values are removed (i.e. NA=0)


##### Missing value imputation 2. Random Forest - Impute missing values using RandomForest from missForest library #####

# load required library
p_load(missForest)

# (Caution! It would take very long time for running! more than 3 hours..(^.^;;;;) ) 

# for whole dataset
# Perform imputation of missing data in a data frame using the Random Forest algorithm. 
secom_data_mF_all <- missForest(secom_data_004_bind) 

# check imputed values 
secom_data_mF_all <- secom_data_mF_all$ximp
head(secom_data_mF_all)
dim(secom_data_mF_all)

# for train dataset
# Perform imputation of missing data in a data frame using the Random Forest algorithm. 
secom_data_mF_all <- missForest(training_set) 

# check imputed values 
secom_data_mF_all <- secom_data_mF_all$ximp
head(secom_data_mF_all)
dim(secom_data_mF_all)

# for test dataset
# Perform imputation of missing data in a data frame using the Random Forest algorithm. 
secom_data_mF_test <- missForest(test_set_knn) 

# check imputed values 
secom_data_mF_test <- secom_data_mF_test$ximp
head(secom_data_mF_test)
dim(secom_data_mF_test)

# checking if there are missing values after Random Forest imputation
sum(is.na(secom_data_mF_all))
# -> As a result, every missing values are removed (i.e. NA=0)

### -> As a result of missing value imputation via knn & Random forest, 
###    every missing values were removed!

####################### 2-1. Feature Selection : Boruta ############################
# Load required library(i.e.Boruta)
p_load(Boruta)

# conducting Boruta for feature selection
set.seed(500)
Boruta.secom_data <- Boruta(Status ~ ., data = secom_data_mF_all, doTrace=2, ntree = 500) 
plot(Boruta.secom_data)
Boruta.secom_data
# As Boruta runned by Random Forest approach, result would be different with each running, but one example could be as like below.
# -> 20 attributes confirmed important: Feature104 etc.
# -> 411 attributes confirmed unimportant: Feature 1 etc.
# -> 8 tentative attributes left: Feature125 etc. 

# As we still have tentative attributes left, let's increase the number of running to 100 -> 150.
set.seed(500)
Boruta.secom_data_expend <- Boruta(Status ~ ., data = training_set_mF, doTrace=2, ntree = 500, maxRuns=150) 
Boruta.secom_data_expend

# As Boruta runned by Random Forest approach, result would be different with each running, but one example could be as like below.
# -> 23 attributes confirmed important: Feature104 etc. 
# -> 412 attributes confirmed unimportant: Feature1 etc.
# -> 4 tentative attributes left: Feature125 etc.

# to fill missing decisions by simple comparision of the median value of Z score
# for handling tentative attributes which are still left
boruta_mF <- TentativeRoughFix(Boruta.secom_data_expend)
boruta_mF

# As Boruta runned by Random Forest approach, result would be different with each running, but one example could be as like below.
# -> Tentatives roughfixed over the last 149 iterations.
# -> 23 attributes confirmed important: Feature104 etc. 
# -> 416 attributes confirmed unimportant: Feature1 etc. 

# check out the finalized selected features as the form of formaular
getConfirmedFormula(Boruta.secom_data_expend)

# check out the finalized result as a table form
attStats(Boruta.secom_data_expend)
df_boruta_knn <- data.frame(attStats(Boruta.secom_data_expend))
df_boruta_knn[20:40,]

# visualization. Plot the feature importance
plot(Boruta.secom_data_expend)

# final feature selection by Boruta  
boruta_mF_001 <- names(boruta_mF$finalDecision[boruta_mF$finalDecision %in% c("Confirmed")]) 
boruta_selected_training_mF <- secom_data_mF_all[,c(boruta_mF_001, "Status")]

# check out the final dimension of the Buruta selected dataset
dim(boruta_selected_training_mF)

################### (Optional) 2-2-1. Data transformation : Yeo-Johnson power transformation #################################
# !! OPTIONAL !! ONLY for making the dataset has linearlity
# which methods request the linearlity or normality of dataset?
# -> features selection : PCA
# -> Models : Regressions (e.g. linear regression, logistic regression, Lasso regression, Ridge regression etc.)

# load livrary
p_load(caret)

# Yeo-Johnson power transformation 
# please choose your data set. Which dataset do you want to use?
# knn imputed datset? or Random Forest imputed dataset?

#### 1) for knn imputed dataset

# As same dataset names reqeuired many time in the algorithm, it would be better to define the name of dataset
# if you want to change the dataset, you can just change the name of dataset once.

# define the required dataset. 
test_set_pt <- secom_data_knn_all

# run the Yoe_Johnson power transformation via preProcess function from caret package
test_set_power <- preProcess(test_set_pt[,-ncol(test_set_pt)], method = c("YeoJohnson"))
test_set_powerT <-predict(test_set_power, test_set_pt[,-ncol(test_set_pt)])
secom_data_all_powerT <- cbind(test_set_powerT, test_set_pt[,ncol(test_set_pt)])
colnames(secom_data_all_powerT)[ncol(test_set_pt)] <- c("Status")

# check out the result fo power transformation
dim(secom_data_all_powerT)
head(secom_data_all_powerT)
table(secom_data_all_powerT$Status)

# check out the lineality of the values. Just for test, please select several names of features. 
# (e.g. feature1, feature2 etc.)
qqnorm(secom_data_all_powerT$Feature1)

#### 2) for Random Forest(missForest) imputed dataset

# define the required dataset. 
test_set_pt <- secom_data_mF_all

test_set_power <- preProcess(test_set_pt[,-ncol(test_set_pt)], method = c("YeoJohnson"))
test_set_powerT <-predict(test_set_power, test_set_pt[,-ncol(test_set_pt)])
secom_data_all_powerT_mF <- cbind(test_set_powerT, test_set_pt[,ncol(test_set_pt)])
colnames(secom_data_all_powerT_mF)[ncol(test_set_pt)] <- c("Status")

# check out the result fo power transformation
dim(secom_data_all_powerT_mF)
head(secom_data_all_powerT_mF)
table(secom_data_all_powerT_mF$Status)

# check out the lineality of the values. Just for test, please select several names of features. 
# (e.g. feature1, feature2 etc.)
qqnorm(secom_data_all_powerT_mF$Feature1)

################ 2-2-2. Feture Reduction : PCA ###################

## Feature Reduction : PCA 
# please choose your data set. Which dataset do you want to use?
# knn imputed datset? or Random Forest imputed dataset?

##### 1) knn imputed dataset

# Feature Reduction : PCA 
secom_data_pca <- prcomp(secom_data_knn_all)

# check out the result of PCA
names(secom_data_pca)
summary(secom_data_pca)
dim(secom_data_pca$x)

# see the result of PCA and decide how many components could be selected
screeplot(secom_data_pca, type = 'l', npcs= length(secom_data_pca$sdev), main="Screeplot",ylim=c(0,1) , col="red")
var <- secom_data_pca$sdev^2
Prop.var<- var/sum(var)
plot(cumsum(Prop.var), col="red", xlab="Principal Component",ylab="Cumulative Proportion of Variance Explained",ylim=c(0,1),type="b")
# -> As a result 64 components explains 80% of variability.

# attach the labels data and rename the last column to "Status"
# 64 components which explains 80% of variability of data were selected
pca_final=as.data.frame(secom_data_pca$x[,1:64])
pca_final=cbind(pca_final,secom_data_mF_all$Status)
colnames(pca_final)[65]=c("Status")

# check out the final PCA results
dim(pca_final)

##### 2) Random Forest(missForest) imputed dataset

# Feature Reduction : PCA 
secom_data_pca_mF <- prcomp(secom_data_mF_all)

# check out the result of PCA
names(secom_data_pca_mF)
summary(secom_data_pca_mF)
dim(secom_data_pca_mF$x)

# see the result of PCA and decide how many components could be selected
screeplot(secom_data_pca_mF, type = 'l', npcs= length(secom_data_pca_mF$sdev), main="Screeplot",ylim=c(0,1) , col="red")
var <- secom_data_pca_mF$sdev^2
Prop.var<- var/sum(var)
plot(cumsum(Prop.var), col="red", xlab="Principal Component",ylab="Cumulative Proportion of Variance Explained",ylim=c(0,1),type="b")
# -> As a result 64 components explains 80% of variability.

# attach the labels data and rename the last column to "Status"
# 64 components which explains 80% of variability of data were selected
pca_final=as.data.frame(secom_data_pca_mF$x[,1:64])
pca_final=cbind(pca_final,secom_data_mF_all$Status)
colnames(pca_final)[65]=c("Status")

# check out the final PCA results
dim(pca_final)


################# 2-2-3. (ONLY for PCA approach!) split the dataset into train and test datasets #################
p_load(caTools)
set.seed(123)

# train: test data set -> 80% : 20%
split = sample.split(pca_final$Status, SplitRatio =0.8)
train_data_pca = subset(pca_final, split == TRUE)
test_data_pca = subset(pca_final, split == FALSE)

# check out the result of data splition
dim(train_data_pca)
dim(test_data_pca)
head(train_data_pca)

################## 3. data Balancing : Applying Sampling methods to make balancing the Imbalanced Data ###################

# As the data has less fail values, we have to apply sampling methods to balance the data
# the best sampling method is SMOTE and SMOTE sampling method using DMwR package
# install.packages('DMwR') for SMOTE sampling approach

# check out the balance of dataset for training/test set
# training set
table(train_data_pca$Status)
# -> as a result, it's totally imbalanced: -1(1170), 1(83)

# load required packages
p_load(data.table)
p_load(psych)
p_load(mvtnorm)
p_load(caret)
p_load(PRROC)
p_load(ggplot2)
p_load(caTools)
p_load(pROC)
p_load(dplyr)
p_load(DMwR)
p_load(ROSE)

########### 3-0. Before sampling ##########

# check out the number of valuables before sampling : Does my dataset balanced or imbalanced?
print('Number of valuables in train dataset before applying sampling methods')
print(table(train_data_pca$Status))
# -> As a result, -1:1170, 1:83 => Totally imbalanced dataset!

# just for visualization : Plot the number of valuables before sampling
df_noSampling <- data.frame(Status=c("-1", "1"),
                            No_Valuables=c(1170, 83))

p_noSampling<-ggplot(df_noSampling, aes(x=Status, y=No_Valuables, fill=Status)) +
  geom_bar(stat="identity")+theme_minimal()
p_noSampling

########### 3-1. Oversampling ##########

# as Fail(1) shows less occurrence, so this Over sampling method will increase the Fail(1) records
# Here sample size is then N = 1170*2 (i.e. 2 times of size of training dataset) 
over_sample_train_data <- ovun.sample(Status ~ ., data = train_data_pca, method="over", N=2340)$data
print('Number of transactions in train dataset after applying Over sampling method')
print(table(over_sample_train_data$Status))
# -> As a result, -1:1170, 1:1170 => become to be balanced!

# just for visualization : Plot the number of valuables before sampling
df_overSampling <- data.frame(Status=c("-1", "1"),
                              No_Valuables=c(1170, 1170))

p_overSampling<-ggplot(df_overSampling, aes(x=Status, y=No_Valuables, fill=Status)) +
  geom_bar(stat="identity")+theme_minimal()
p_overSampling

########### 3-2. undersampling ##########
# Undersampling,as Fail(1) are having less occurrence, 
# so this Under sampling method will descrease the Good records untill matches Fraud records, 
# But, you see that we’ve lost significant information from the sample. 
under_sample_train_data <- ovun.sample(Status ~ ., data = train_data_pca, method="under", N=166)$data
print('Number of transactions in train dataset after applying Under sampling method')
print(table(under_sample_train_data$Status))
# -> As a result, -1:83, 1:83 => become to be balanced!

# just for visualization : Plot the number of valuables before sampling
df_underSampling <- data.frame(Status=c("-1", "1"),
                               No_Valuables=c(83, 83))

p_underSampling<-ggplot(df_underSampling, aes(x=Status, y=No_Valuables, fill=Status)) +
  geom_bar(stat="identity")+theme_minimal() + coord_cartesian(ylim = c(0, 1170))
p_underSampling

########### 3-3. Hybrid(Mixed) sampling ##########

# Mixed Sampling, apply both under sampling and over sampling on this imbalanced data 
both_sample_train_data <- ovun.sample(Status ~ ., data = train_data_pca, method="both", p=0.5, seed=222, N=1253)$data
print('Number of transactions in train dataset after applying Mixed sampling method')
print(table(both_sample_train_data$Status))
# -> As a result, -1:1165, 1:1175 => become to be almost balanced!

# just for visualization : Plot the number of valuables before sampling
df_bothSampling <- data.frame(Status=c("-1", "1"),
                              No_Valuables=c(1165, 1175))

p_bothSampling<-ggplot(df_bothSampling, aes(x=Status, y=No_Valuables, fill=Status)) +
  geom_bar(stat="identity")+theme_minimal()
p_bothSampling

########### 3-4. ROSE sampling ##########

# ROSE Sampling, this helps us to generate data synthetically. It generates artificial datas instead of dulicate data.
rose_sample_train_data <- ROSE(Status ~ ., data = train_data_pca,  seed=111)$data
print('Number of transactions in train dataset after applying ROSE sampling method')
print(table(rose_sample_train_data$Status))
# -> As a result, -1:625, 1:628 => become to be almost balanced!

# just for visualization : Plot the number of valuables before sampling
df_roseSampling <- data.frame(Status=c("-1", "1"),
                              No_Valuables=c(625, 628))

p_roseSampling<-ggplot(df_roseSampling, aes(x=Status, y=No_Valuables, fill=Status)) +
  geom_bar(stat="identity")+theme_minimal() + coord_cartesian(ylim = c(0, 1170))
p_roseSampling

########### 3-5. SMOTE sampling ##########

# SMOTE(Synthetic Minority Over-sampling Technique) Sampling
# formula - relates how our dependent variable acts based on other independent variable.
# data - input data
# perc.over - controls the size of Minority class
# perc.under - controls the size of Majority class
# since my data has less Majority class, increasing it with 200 and keeping the minority class to 100.
train_data_pca$Status = as.factor(train_data_pca$Status) # to make the type of Status as a factor : requirement for SMOTE function
train_data_pca_smote <- SMOTE(Status ~ ., data = train_data_pca, perc.over = 1400, perc.under=100)
print('Number of transactions in train dataset after applying SMOTE sampling method')
print(table(train_data_pca_smote$Status))
# -> As a result, -1:1162, 1:1245 => become to be balanced!

# just for visualization : Plot the number of valuables before sampling
df_smoteSampling <- data.frame(Status=c("-1", "1"),
                               No_Valuables=c(166, 166))

p_smoteSampling<-ggplot(df_smoteSampling, aes(x=Status, y=No_Valuables, fill=Status)) +
  geom_bar(stat="identity")+theme_minimal() + coord_cartesian(ylim = c(0, 1170))
p_smoteSampling

##################### 4. Model builing. 1) SVM classifier #####################

# choose your preferred sampled datset. (e.g. smote sampled, rose sampled..etc.)
training_data_svm <- train_data_pca_smote

# Encoding the target feature as factor 
training_data_svm$Status = factor(training_data_svm$Status, levels = c(-1, 1)) 

### Fitting SVM model to the Training set 

# load required library
p_load(e1071) 

##### 3 different ways of SVM model. Please choose the one SVM model(linear/polynomial/Radial kernel)

# for svm linear
smote_svm_classifier_pca_lin <- svm(formula = Status ~ ., 
                                       data = training_data_svm, 
                                       type = 'C-classification', 
                                       kernel = 'linear') 
smote_svm_classifier_pca_lin

# for svm polynomial
smote_svm_classifier_boruta_poly <- svm(formula = Status ~ ., 
                                           data = training_data_svm, 
                                           type = 'C-classification', 
                                           kernel = 'polynomial') 
smote_svm_classifier_boruta_poly

# for svm Radial kernel
Radial_svm_classifier_smote =svm(Status ~ .,data=training_data_svm, kernel="radial",cost=5,scale=F)
Radial_svm_classifier_smote

### Predicting with the Test set results 

# define the test dataset
test_dataset_svm <- test_data_pca

# svm_linear
svm_pred_pca_powerT_lin = predict(smote_svm_classifier_pca_lin, newdata = test_dataset_svm[-ncol(test_dataset_svm)]) 
# svm_polinomiar
svm_pred_boruta_knn_poly = predict(smote_svm_classifier_boruta_poly, newdata = test_dataset_svm[-ncol(test_dataset_svm)]) 
# for svm Radial kernel
svm_pred_boruta_knn_kernel = predict(Radial_svm_classifier_smote, newdata = test_dataset_svm[-ncol(test_dataset_svm)]) 

### Making the Confusion Matrix 

# for SVM linear
table(test_dataset_svm[, ncol(test_dataset_svm)], svm_pred_pca_powerT_lin) 
# for poly
table(test_dataset_svm[, ncol(test_dataset_svm)], svm_pred_boruta_knn_poly) 
# for Radial kernel
table(test_dataset_svm[, ncol(test_dataset_svm)], svm_pred_boruta_knn_kernel) 

##### Evaluation : ROC curve of over sampling data

# for SVM linear
roc.curve(test_dataset_svm$Status, svm_pred_pca_powerT_lin, plotit = T, main ="ROC model")
print(roc.curve(test_dataset_svm[, ncol(test_dataset_svm)], svm_pred_pca_powerT_lin))
confusionMatrix(table(test_dataset_svm[, ncol(test_dataset_svm)], svm_pred_pca_powerT_lin))

# for SVM polynomiar
roc.curve(test_dataset_svm$Status, svm_pred_boruta_knn_poly, plotit = T, main ="ROC model")
print(roc.curve(test_dataset_svm[, ncol(test_dataset_svm)], svm_pred_boruta_knn_poly))
confusionMatrix(table(test_dataset_svm[, ncol(test_dataset_svm)], svm_pred_boruta_knn_poly))

# for SVM Radial Kernel
roc.curve(test_dataset_svm$Status, svm_pred_boruta_knn_kernel, plotit = T, main ="ROC model")
print(roc.curve(test_dataset_svm[, ncol(test_dataset_svm)], svm_pred_boruta_knn_kernel))
confusionMatrix(table(test_dataset_svm[, ncol(test_dataset_svm)], svm_pred_boruta_knn_kernel))

##################### 4. Model builing. 2) Naive Bayse #####################

# load required package
p_load(e1071)
set.seed(100)

# define the dataset
train_data_Naive <- rose_sample_train_data

# set the naive bayse classifier 
naive_fit_pca<-naiveBayes(formula = Status~., data=train_data_Naive, type="class")

# check out the resukt of Naive Bayse classifier
summary(naive_fit_pca)
summary(pca_svm_classifier_rose)
summary(naive_fit_boruta_knn_rose)
summary(naive_fit_pca_mf_smote)

# define the test set
pca_mf_test <- test_data_pca

# prediciton with test set
naive_pre_pca_mf_rose <- predict(naive_fit_pca_mf_rose, pca_mf_test[-ncol(pca_mf_test)])

### confusionMatrix
confusionMatrix(table(naive_pre_pca_mf_rose, pca_mf_test$Status))

### ROC curve
roc.curve(as.factor(pca_mf_test$Status), naive_pre_pca_mf_rose)

##################### 4. Model builing. 3) Random Forest Model #####################
# load the required package
p_load(randomForest)

# Create a Random Forest model with default parameters
model_SMOTE_001 <- randomForest(Status ~ ., data = smote_sample_train_data, importance = TRUE)
model_SMOTE_001

# Fine tuning parameters of Random Forest model
model_SMOTE_002 <- randomForest(Status ~ ., data = smote_sample_train_data, ntree = 500, mtry = 6, importance = TRUE)
model_SMOTE_002


# Predicting with the test data
predValid <- predict(model_SMOTE_002, secom_data_mF_test, type = "class")

### confusionMatrix
confusionMatrix(table(predValid, secom_data_mF_test$Status))

### ROC curve
roc.curve(as.factor(secom_data_mF_test$Status), predValid)

##################### 4. Model builing. 4) Decision Tree #####################

# load required packages
# tree packages could be installed at least over R Ver. 3.6.0
p_load(tree)
p_load(BiocManager)
set.seed(1)

# define the dataset
train_data_decisionT <- smote_sample_train_data

# set the Decision Tree classifier 
tree_fit_knn_smote <- tree(Status~., data=train_data_decisionT)

# check out the result of Desicion Tree classifier
plot(tree_fit_knn_smote)
text(tree_fit_knn_smote)

# define the test set
pca_mf_test <- secom_data_mF_test

# prediction with the test set
tree_pre_boruta_mF_smote <- predict(tree_fit_knn_smote, pca_mf_test[-ncol(pca_mf_test)], type="class")

### confusionMatrix
confusionMatrix(table(tree_pre_boruta_mF_smote, as.factor(pca_mf_test$Status)))

### ROC curve
roc.curve(as.factor(pca_mf_test$Status), tree_fit_knn_smote)

##################### 4. Model builing. 5) Regression #####################

############ 4.5-1) Lasso regression 
# Power transformation would be applied before the regression upon the reqruirement
# train data : secom_data_all_powerT
# test data : test_set_powerT

# load reqruired package
p_load(glmnet)
set.seed(500)

# fit the lasso regression model
sh<-10^seq(10,-2,length=100)
x_smote <- model.matrix(Status ~ ., data=secom_data_all_powerT)[,-1]
y_smote <- secom_data_all_powerT$Status
# cross validation to findout minimum lamda(*alpha=0:Ridge regression, alpha=1:lasso regression)
cv.lasso <-cv.glmnet(x_smote, y_smote, alpha=1, family = "binomial") 
plot(cv.lasso, main = "Lasso, SMOTE, powerT")
# put minimum lamda to make best lasso
bestlam.lasso <-cv.lasso$lambda.min 
# build a best lasso model
smote_classifier_lasso_mF_powerT <-glmnet(x_smote, y_smote, alpha=1,lambda=sh, family = "binomial") 
# draw plot of training MSE as a function of lambda
plot(smote_classifier_lasso_mF_powerT) 

# Use best lambda to predict test data
x_test <- model.matrix(Status ~ ., data=test_set_powerT)[,-1]
smote_lasso_pred_mF = predict(smote_classifier_lasso_mF_powerT, s = bestlam.lasso, newx = x_test, type="response") 
y_pred_smote_mF = ifelse(smote_lasso_pred_mF>0.5, 1, -1)

### confusionMatrix
confusionMatrix(table(y_pred_smote_mF, as.factor(test_set_powerT$Status)))

### ROC curve
roc.curve(as.factor(test_set_powerT$Status), y_pred_smote_mF)

############ 4.5-2) Ridge Regression
# RIDGE regression - SMOTE sampling, RandomForest imputed dataset
x_smote_rid <- model.matrix(Status ~ ., data=secom_data_all_powerT)[,-1]
y_smote_rid <- secom_data_all_powerT$Status

#Find the optimal lambda value via cross validation
cv.out=cv.glmnet(x_smote_rid, y_smote_rid,alpha=0, family = "binomial")
bestlam=cv.out$lambda.min

#Fit a ridge regression model
grid=10^seq(10,-2,length=100)
smote_classifier_ridge_mF=glmnet(x_smote_rid, y_smote_rid, alpha = 0, lambda=grid, family = "binomial")

#Compute the test error
smote_ridge_pred_mF=predict(smote_classifier_ridge_mF,s=bestlam,newx=x_test)
y_pred_smote_mF = ifelse(smote_ridge_pred_mF>0.5, 1, -1)

#Compute the MSE for ridge regression
ridge.mse2=mean((ridge.pred-y[-train])^2)

#Display Ridge Coeffcieints
ridge.coef2=predict(ridge.mod,type="coefficients",s=bestlam)

### confusionMatrix
confusionMatrix(table(y_pred_smote_mF, as.factor(test_set_powerT$Status)))

### ROC curve
roc.curve(as.factor(test_set_powerT$Status), y_pred_smote_mF)

############ 4.5-3) Logostic Regression

# fit the logistic regression model
SMOTE_model_001  <- glm(Status ~.,family=binomial(link='logit'), data= secom_data_all_powerT)

# prediction with test set
fitted.results_pre <- predict(SMOTE_model_001,newdata= test_set_powerT,type='response')
fitted.results <- ifelse(fitted.results_pred > 0.5,1,-1)

# check out the result of logistic regression and see the error rate
misClasificError <- mean(fitted.results != test_data_missForest_imputed$Status)
print(paste('Accuracy',1-misClasificError))

### confusionMatrix
confusionMatrix(table(fitted.results, as.factor(test_set_powerT$Status)))

### ROC curve
roc.curve(as.factor(test_set_powerT$Status), fitted.results)

################### (Optional) Visualization: Plotting ROC curve at onece ###########################

# to use the combined ROC curve, you should put the selected factors to the below
# ROC curve of best model 
test_data_eval <- test_data_pca
pred_test_model <- smote_svm_pred_pca_powerT
roc.curve(as.factor(test_data_eval$Status), pred_test_model, plotit = T, main ="ROC model : RF impt > YJ powerT > PCA > SMOTE sample > SVM linear")
print(roc.curve(test_data_eval$Status, pred_test_model))
confusionMatrix(table(test_data_eval[, ncol(test_data_eval)], pred_test_model))
dim(test_data_eval)

### Plotting ROC curve at onece
p_load(ROCR)
pr1<-prediction(y_pred_smote, test_set_sel$Status)
pr2<-prediction(y_pred_rose, test_set_sel$Status)
pr3<-prediction(y_pred_both, test_set_sel$Status)
pr4<-prediction(y_pred_over, test_set_sel$Status)
pr5<-prediction(y_pred_under, test_set_sel$Status)
perf1<- performance(pr1,"tpr","fpr")
perf2<- performance(pr2,"tpr","fpr")
perf3<- performance(pr3,"tpr","fpr")
perf4<- performance(pr4,"tpr","fpr")
perf5<- performance(pr5,"tpr","fpr")

plot(perf1, col='red')
plot(perf2,add = TRUE, col='green')
plot(perf3,add = TRUE, col='blue')
plot(perf4,add = TRUE, col='black')
plot(perf5, add= TRUE, col='orange')


