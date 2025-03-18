setwd('~/Documenti/MATLAB/biomarker/homework/R')

###### complete data ##############################
# PREDICTION WITH COMPLETE DATASET

library(readr)
Rdata<- read_csv("Radata.csv")

Rdata$disease=1
Rdata$disease[34:53]=0
Rdata$disease=as.factor(Rdata$disease)
summary(Rdata)

set.seed(123)
# creates balanced trn/tst
library(caret)
idx <- createDataPartition(Rdata$disease, p = .7, list = FALSE, times = 1) 
trn <- Rdata[idx,]
tst <- Rdata[-idx,]


### RANDOM FOREST ########

# first with train for tuning parameter: mtry
train.control = trainControl(method = "LOOCV")
set.seed(123)
RFF <- train(disease~., data=trn, tuneLength=7, tuneGrid=expand.grid(.mtry = (2:8)),
             method="rf", trControl = train.control)
RFF
pred.RFF <- predict(RFF,tst)
confusionMatrix(pred.RFF,tst$disease) # perfect

### SVM WITH TRAIN ########

# radial kernel
svm.rad.cv <- train(disease~., data=trn, method="svmRadial", trControl=train.control,
                    preProcess=c("center","scale"), tuneGrid=expand.grid(C=c(0.2,0.5,0.8,1,2,3,4),
                                                                         sigma=c(0.0001,0.001,0.01,0.1,0.5)))

# polinomial kernel
svm.pol.cv <- train(disease~., data=trn, method="svmPoly", trControl=train.control,
                    preProcess=c("center","scale"), tuneGrid=expand.grid(C=c(0.001,0.01, 0.1,1,10,100),
                                                                         degree=c(1,2,3,4), scale=1))

pred.rad<-predict(svm.rad.cv,tst)
pred.pol<-predict(svm.pol.cv,tst)

confusionMatrix(pred.rad,tst$disease) # perfect
confusionMatrix(pred.pol,tst$disease) # perfect

# we can see a perfect accuracy in several model when we use complete dataset

## Logisti regression
lr8 <- train(disease~., method="glm", family=binomial, data=trn,
              trControl =train.control,preProcess = c("center","scale"))

pred.lr8 <- predict(lr8, tst)
confusionMatrix(pred.lr8,tst$disease, positive="1")

library(pROC)
pred.lr8 <- predict(lr8, tst,type = "prob")
myroc8 <- roc(tst$disease ~  pred.lr8[,2],  plot=T)
#Accuracy : 0.6 
#Kappa : 0.2105
#Sensitivity : 0.5556          
#Specificity : 0.6667 
# can see that feature lose power prediction of glm model because of the feature have high correlation each other 


####### selection features  (48 fetures)(VIF) ########################################
#PREDICTION WITH SELCTION FEATURES (first step)

library(readr)
library(caret)
library(pROC)
Rdata48<- read_csv("Rdata48.csv")

Rdata48$disease=1
Rdata48$disease[34:53]=0
Rdata48$disease=as.factor(Rdata48$disease)

# Configuration
train.control = trainControl(method = "LOOCV")
seeds <- c(123, 456, 1011, 1213, 1415,617, 1819, 2021,12223,789 )

### LOGISTIC MODEL ########

# build a structure to save data
accuraciesGLM48 <- vector("numeric", length = length(seeds))
confusion_matricesGLM48 <- list()
sensitivitiesGLM48<- vector("numeric", length = length(seeds))
specificitiesGLM48 <- vector("numeric", length = length(seeds))
kappaGLM48 <- vector("numeric", length = length(seeds))

for (i in 1:length(seeds)) {
  set.seed(seeds[i])
  
  # creates balanced trn/tst
  idx <- createDataPartition(Rdata48$disease, p = .7, list = FALSE, times = 1) 
  trn48 <- Rdata48[idx,]
  tst48 <- Rdata48[-idx,]
  
  #model
  lr48 <- train(disease~., method="glm", family=binomial, data=trn48,
               trControl =train.control,preProcess = c("center","scale"))
  
  pred.lr48 <- predict(lr48, tst48)
  cm<-confusionMatrix(pred.lr48,tst48$disease, positive="1")
  
  #ROC plot
  pred.lr48 <- predict(lr48, tst48,type="prob")
  myroc48 <- roc(tst48$disease ~  pred.lr48[,2],  plot=T, print.auc=T)
  
  accuracy <- cm$overall["Accuracy"]
  kappa<-cm$overall["Kappa"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  
  # save result
  accuraciesGLM48[i] <- accuracy
  confusion_matricesGLM48[[i]] <- cm$table
  sensitivitiesGLM48[i] <- sensitivity
  specificitiesGLM48[i] <- specificity
  kappaGLM48[i]<-kappa
}

#print model result
for (i in 1:length(seeds)) {
  cat("Modello", i, "\n")
  cat("Accuracy:", accuraciesGLM48[i], "\n")
  cat("Confusion Matrix:\n")
  print(confusion_matricesGLM48[[i]])
  cat("Sensitivity:", sensitivitiesGLM48[i], "\n")
  cat("Specificity:", specificitiesGLM48[i], "\n")
  cat("kappa:",kappaGLM48[i],"\n")
  cat("\n")
}

#accuracy between 0.33-0.93, depending from seeds
#sensitivity from 0.222 a 0.888
# specificity 0.333 a 1
# kappa -0.25 a 0.864

library(pROC)
pred.lr48 <- predict(lr48, tst48,type="prob")
myroc48 <- roc(tst48$disease ~  pred.lr48[,2],  plot=T)

### SVM polinomial kernel ######

# build a structure to save data
accuraciesSVM.P48 <- vector("numeric", length = length(seeds))
confusion_matricesSVM.P48 <- list()
sensitivitiesSVM.P48<- vector("numeric", length = length(seeds))
specificitiesSVM.P48 <- vector("numeric", length = length(seeds))
kappaSVM.P48 <- vector("numeric", length = length(seeds))

for (i in 1:length(seeds)) {
  set.seed(seeds[i])
  
  # creates balanced trn/tst
  idx <- createDataPartition(Rdata48$disease, p = .7, list = FALSE, times = 1) 
  trn48 <- Rdata48[idx,]
  tst48 <- Rdata48[-idx,]
  
  #model
  svm.pol.48 <- train(disease~., data=trn48, method="svmPoly", trControl=train.control,
                     preProcess=c("center","scale"), 
                     tuneGrid=expand.grid(C=c(0.001,0.01, 0.1,1,10,100),
                                          degree=c(1,2,3,4), scale=1))
  
  pred.pol48<-predict(svm.pol.48,tst48) 
  cm <-confusionMatrix(pred.pol48,tst48$disease) 
  
  accuracy <- cm$overall["Accuracy"]
  kappa<-cm$overall["Kappa"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  
  # save result
  accuraciesSVM.P48[i] <- accuracy
  confusion_matricesSVM.P48[[i]] <- cm$table
  sensitivitiesSVM.P48[i] <- sensitivity
  specificitiesSVM.P48[i] <- specificity
  kappaSVM.P48[i]<-kappa
}

#print model result
for (i in 1:length(seeds)) {
  cat("Modello", i, "\n")
  cat("Accuracy:", accuraciesSVM.P48[i], "\n")
  cat("Confusion Matrix:\n")
  print(confusion_matricesSVM.P48[[i]])
  cat("Sensitivity:", sensitivitiesSVM.P48[i], "\n")
  cat("Specificity:", specificitiesSVM.P48[i], "\n")
  cat("kappa:",kappaSVM.P48[i],"\n")
  cat("\n")
}
# Accuracy: da 0.7333333 a 1
# Specificity: da 0.5555556 a 1
# Kappa: da 0.5 a 1
# Sensibilità: da 0.8333333 a 1

### RANDOM FOREST ########

# build a structure to save data
accuraciesRFF48 <- vector("numeric", length = length(seeds))
confusion_matricesRFF48 <- list()
sensitivitiesRFF48 <- vector("numeric", length = length(seeds))
specificitiesRFF48 <- vector("numeric", length = length(seeds))
kappaRFF48<- vector("numeric", length = length(seeds))

for (i in 1:length(seeds)) {
  set.seed(seeds[i])
  
  # creates balanced trn/tst
  idx <- createDataPartition(Rdata48$disease, p = .7, list = FALSE, times = 1) 
  trn48 <- Rdata48[idx,]
  tst48 <- Rdata48[-idx,]
  
  # model
  RFF48 <- train(disease~., data=trn48, tuneLength=7,
                method="rf", trControl = train.control)
  RFF48
  pred.RFF48 <- predict(RFF48,tst48)
  
  cm <-confusionMatrix(pred.RFF48,tst48$disease)
  
  accuracy <- cm$overall["Accuracy"]
  kappa<-cm$overall["Kappa"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  
  # save result
  accuraciesRFF48[i] <- accuracy
  confusion_matricesRFF48[[i]] <- cm$table
  sensitivitiesRFF48[i] <- sensitivity
  specificitiesRFF48[i] <- specificity
  kappaRFF48[i]<-kappa
}

#print model result
for (i in 1:length(seeds)) {
  cat("Modello", i, "\n")
  cat("Accuracy:", accuraciesRFF48[i], "\n")
  cat("Confusion Matrix:\n")
  print(confusion_matricesRFF48[[i]])
  cat("Sensitivity:", sensitivitiesRFF48[i], "\n")
  cat("Specificity:", specificitiesRFF48[i], "\n")
  cat("kappa:",kappaRFF48[i],"\n")
  cat("\n")
}

#Accuracy: 0.8 a 1 
#Sensitivity: 0.833 a 1 
#Specificity: 0.666 a 1, 
#kappa: 0.594 a 1

####### selection features  (13 features)(VIF) ########################################
#PREDICTION WITH SELCTION FEATURES (second step)

library(readr)
Rdata1 <- read_csv("Rdata13.csv")

Rdata1$disease=1
Rdata1$disease[34:53]=0
Rdata1$disease=as.factor(Rdata1$disease)
summary(Rdata1)


# Configuration
train.control = trainControl(method = "LOOCV")
seeds <- c(123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021, 2223)

### RANDOM FOREST ########

# build a structure to save data
accuraciesRFF1 <- vector("numeric", length = length(seeds))
confusion_matricesRFF1 <- list()
sensitivitiesRFF1 <- vector("numeric", length = length(seeds))
specificitiesRFF1 <- vector("numeric", length = length(seeds))
kappaRFF1<- vector("numeric", length = length(seeds))

for (i in 1:length(seeds)) {
  set.seed(seeds[i])
  
  #creates balanced trn/tst
  idx <- createDataPartition(Rdata1$disease, p = .7, list = FALSE, times = 1) 
  trn1 <- Rdata1[idx,]
  tst1 <- Rdata1[-idx,]
  
  
  # model
  RFF1 <- train(disease~., data=trn1, tuneLength=7,
                method="rf", trControl = train.control)
  RFF1
  pred.RFF1 <- predict(RFF1,tst1)
  
  cm <-confusionMatrix(pred.RFF1,tst1$disease)
  
  accuracy <- cm$overall["Accuracy"]
  kappa<-cm$overall["Kappa"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  
  # save result
  accuraciesRFF1[i] <- accuracy
  confusion_matricesRFF1[[i]] <- cm$table
  sensitivitiesRFF1[i] <- sensitivity
  specificitiesRFF1[i] <- specificity
  kappaRFF1[i]<-kappa
}

#print model result
for (i in 1:length(seeds)) {
  cat("Modello", i, "\n")
  cat("Accuracy:", accuraciesRFF1[i], "\n")
  cat("Confusion Matrix:\n")
  print(confusion_matricesRFF1[[i]])
  cat("Sensitivity:", sensitivitiesRFF1[i], "\n")
  cat("Specificity:", specificitiesRFF1[i], "\n")
  cat("kappa:",kappaRFF1[i],"\n")
  cat("\n")
}

#Accuracy: major of the models have accuracy equal to 1, the other have accurancy more than 0.86 
#Sensitivity: 1  
#Specificity: 1 
#kappa: 1 

### KNN WITH TRAIN #########

# build a structure to save data
accuraciesKNN1 <- vector("numeric", length = length(seeds))
confusion_matricesKNN1 <- list()
sensitivitiesKNN1 <- vector("numeric", length = length(seeds))
specificitiesKNN1 <- vector("numeric", length = length(seeds))
kappaKNN1 <- vector("numeric", length = length(seeds))

for (i in 1:length(seeds)) {
  set.seed(seeds[i])
  
  #creates balanced trn/tst
  idx <- createDataPartition(Rdata1$disease, p = .7, list = FALSE, times = 1) 
  trn1 <- Rdata1[idx,]
  tst1 <- Rdata1[-idx,]
  
  #model
  knnFit1 <- train(disease ~ .,
                   data=trn1,
                   method = "knn",
                   trControl = train.control,
                   preProcess = c("center","scale"),
                   tuneLength = 20)
  # test
  pred_knn1 <- predict(knnFit1,tst1)
  cm <-confusionMatrix(pred_knn1,tst1$disease)
  
  accuracy <- cm$overall["Accuracy"]
  kappa<-cm$overall["Kappa"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  
  # save result
  accuraciesKNN1[i] <- accuracy
  confusion_matricesKNN1[[i]] <- cm$table
  sensitivitiesKNN1[i] <- sensitivity
  specificitiesKNN1[i] <- specificity
  kappaKNN1[i]<-kappa
}

#print model result
for (i in 1:length(seeds)) {
  cat("Modello", i, "\n")
  cat("Accuracy:", accuraciesKNN1[i], "\n")
  cat("Confusion Matrix:\n")
  print(confusion_matricesKNN1[[i]])
  cat("Sensitivity:", sensitivitiesKNN1[i], "\n")
  cat("Specificity:", specificitiesKNN1[i], "\n")
  cat("kappa:",kappaKNN1[i],"\n")
  cat("\n")
}

#Accuracy:  0,8666667 a 1    
#Sensitivity: 1   
#Specificity: 0,7777778 a 1. 
#kappa: 0.8648649 a 1     

#however, it is important to note that the results of the KNN models appear to be
#less stable and consistent than the Random Forest models described earlier

#### SVM WITH TRAIN ##########

## SVM radial kernel ##

# build a structure to save data
accuraciesSVM.R1 <- vector("numeric", length = length(seeds))
confusion_matricesSVM.R1 <- list()
sensitivitiesSVM.R1<- vector("numeric", length = length(seeds))
specificitiesSVM.R1 <- vector("numeric", length = length(seeds))
kappaSVM.R1 <- vector("numeric", length = length(seeds))

for (i in 1:length(seeds)) {
  set.seed(seeds[i])
  
  #creates balanced trn/tst
  idx <- createDataPartition(Rdata1$disease, p = .7, list = FALSE, times = 1) 
  trn1 <- Rdata1[idx,]
  tst1 <- Rdata1[-idx,]
  
  
  #model
  svm.rad.1 <- train(disease~., data=trn1, method="svmRadial", trControl=train.control,
                     preProcess=c("center","scale"),
                     tuneGrid=expand.grid(C=c(0.2,0.5,0.8,1,2,3,4),
                                          sigma=c(0.0001,0.001,0.01,0.1,0.5)))
  
  pred.rad1<-predict(svm.rad.1,tst1)
  cm <-confusionMatrix(pred.rad1,tst1$disease) 
  
  accuracy <- cm$overall["Accuracy"]
  kappa<-cm$overall["Kappa"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  
  # save result
  accuraciesSVM.R1[i] <- accuracy
  confusion_matricesSVM.R1[[i]] <- cm$table
  sensitivitiesSVM.R1[i] <- sensitivity
  specificitiesSVM.R1[i] <- specificity
  kappaSVM.R1[i]<-kappa
}

#print model result
for (i in 1:length(seeds)) {
  cat("Modello", i, "\n")
  cat("Accuracy:", accuraciesSVM.R1[i], "\n")
  cat("Confusion Matrix:\n")
  print(confusion_matricesSVM.R1[[i]])
  cat("Sensitivity:", sensitivitiesSVM.R1[i], "\n")
  cat("Specificity:", specificitiesSVM.R1[i], "\n")
  cat("kappa:",kappaSVM.R1[i],"\n")
  cat("\n")
}

#Accuracy: 0.8666667 e 1
#Sensitivity:  0.8333333 e 1, 
#Specificity: 0.8888889 e 1 
#kappa: 0.7222222 a 1


## SVM polinomial kernel ##

# build a structure to save data
accuraciesSVM.P1 <- vector("numeric", length = length(seeds))
confusion_matricesSVM.P1 <- list()
sensitivitiesSVM.P1<- vector("numeric", length = length(seeds))
specificitiesSVM.P1 <- vector("numeric", length = length(seeds))
kappaSVM.P1 <- vector("numeric", length = length(seeds))

for (i in 1:length(seeds)) {
  set.seed(seeds[i])
  
  #creates balanced trn/tst
  idx <- createDataPartition(Rdata1$disease, p = .7, list = FALSE, times = 1) 
  trn1 <- Rdata1[idx,]
  tst1 <- Rdata1[-idx,]
  
  
  #model
  svm.pol.1 <- train(disease~., data=trn1, method="svmPoly", trControl=train.control,
                     preProcess=c("center","scale"), 
                     tuneGrid=expand.grid(C=c(0.001,0.01, 0.1,1,10,100),
                                          degree=c(1,2,3,4), scale=1))
  
  pred.pol1<-predict(svm.pol.1,tst1) 
  cm <-confusionMatrix(pred.pol1,tst1$disease) 
  
  accuracy <- cm$overall["Accuracy"]
  kappa<-cm$overall["Kappa"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  
  # save result
  accuraciesSVM.P1[i] <- accuracy
  confusion_matricesSVM.P1[[i]] <- cm$table
  sensitivitiesSVM.P1[i] <- sensitivity
  specificitiesSVM.P1[i] <- specificity
  kappaSVM.P1[i]<-kappa
}

#print model result
for (i in 1:length(seeds)) {
  cat("Modello", i, "\n")
  cat("Accuracy:", accuraciesSVM.P1[i], "\n")
  cat("Confusion Matrix:\n")
  print(confusion_matricesSVM.P1[[i]])
  cat("Sensitivity:", sensitivitiesSVM.P1[i], "\n")
  cat("Specificity:", specificitiesSVM.P1[i], "\n")
  cat("kappa:",kappaSVM.P1[i],"\n")
  cat("\n")
}

#Accuracy:  major of the model are 1
#Sensitivity: 1
#Specificity: 1
#kappa: 1 
# but some of these are accuracy more than 0.86 

### LOGISTIC MODEL ########

# build a structure to save data
accuraciesGLM1 <- vector("numeric", length = length(seeds))
confusion_matricesGLM1 <- list()
sensitivitiesGLM1<- vector("numeric", length = length(seeds))
specificitiesGLM1 <- vector("numeric", length = length(seeds))
kappaGLM1 <- vector("numeric", length = length(seeds))

for (i in 1:length(seeds)) {
  set.seed(seeds[i])
  
  #creates balanced trn/tst
  idx <- createDataPartition(Rdata1$disease, p = .7, list = FALSE, times = 1) 
  trn1 <- Rdata1[idx,]
  tst1 <- Rdata1[-idx,]
  
  
  #model
  lr1 <- train(disease~., method="glm",
               family=binomial, data=trn1, 
               trControl =train.control,
               preProcess = c("center","scale"))
  
  pred.lr1 <- predict(lr1, tst1)
  cm <-confusionMatrix(pred.lr1,tst1$disease, positive="1") 
  
  #ROC plot
  pred.lr1 <- predict(lr1, tst1,type = "prob")
  myroc13 <- roc(tst1$disease ~  pred.lr1[,2],  plot=T, print.auc=T)
  
  accuracy <- cm$overall["Accuracy"]
  kappa<-cm$overall["Kappa"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  
  # save result
  accuraciesGLM1[i] <- accuracy
  confusion_matricesGLM1[[i]] <- cm$table
  sensitivitiesGLM1[i] <- sensitivity
  specificitiesGLM1[i] <- specificity
  kappaGLM1[i]<-kappa
}

#print model result
for (i in 1:length(seeds)) {
  cat("Modello", i, "\n")
  cat("Accuracy:", accuraciesGLM1[i], "\n")
  cat("Confusion Matrix:\n")
  print(confusion_matricesGLM1[[i]])
  cat("Sensitivity:", sensitivitiesGLM1[i], "\n")
  cat("Specificity:", specificitiesGLM1[i], "\n")
  cat("kappa:",kappaGLM1[i],"\n")
  cat("\n")
}
#Accuracy: 0.8666667 e 1
#Sensitivity: 1 
#Specificity:  0.7777778 e 1 
#kappa: 0.7368421 a 1

library(pROC)
pred.lr1 <- predict(lr1, tst1,type = "prob")
myroc13 <- roc(tst1$disease ~  pred.lr1[,2],  plot=T)

###### CON FEATURES LASSO #######################################################

library(caret)
library(readr)
Rdata3<- read_csv("Rdataset.csv")

# build a new column of disease
Rdata3$disease=1
Rdata3$disease[34:53]=0
Rdata3$disease=as.factor(Rdata3$disease)
summary(Rdata3)

# test and train
idx <- createDataPartition(Rdata3$disease, p = .7, list = FALSE, times = 1) 
trn3 <- Rdata3[idx,]
tst3 <- Rdata3[-idx,]

#configuration
train.control = trainControl(method = "LOOCV")
seeds <- c(123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021, 2223)


##### RANDOM FOREST #######

# build a structure to save data
accuraciesRFF3 <- vector("numeric", length = length(seeds))
confusion_matricesRFF3 <- list()
sensitivitiesRFF3 <- vector("numeric", length = length(seeds))
specificitiesRFF3 <- vector("numeric", length = length(seeds))
kappaRFF3 <- vector("numeric", length = length(seeds))

for (i in 1:length(seeds)) {
  set.seed(seeds[i])
  
  # test and train
  idx <- createDataPartition(Rdata3$disease, p = .7, list = FALSE, times = 1) 
  trn3 <- Rdata3[idx,]
  tst3 <- Rdata3[-idx,]
  
  # Random Forest model
  RFF3 <- train(disease~., data=trn3, tuneLength=7,
                method="rf", trControl = train.control)
  RFF3
  pred.RFF3 <- predict(RFF3,tst3)
  
  cm <-confusionMatrix(pred.RFF3,tst3$disease)
  
  accuracy <- cm$overall["Accuracy"]
  kappa<-cm$overall["Kappa"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  
  # save result
  accuraciesRFF3[i] <- accuracy
  confusion_matricesRFF3[[i]] <- cm$table
  sensitivitiesRFF3[i] <- sensitivity
  specificitiesRFF3[i] <- specificity
  kappaRFF3[i]<-kappa
}

#print model result
for (i in 1:length(seeds)) {
  cat("Modello", i, "\n")
  cat("Accuracy:", accuraciesRFF3[i], "\n")
  cat("Confusion Matrix:\n")
  print(confusion_matricesRFF3[[i]])
  cat("Sensitivity:", sensitivitiesRFF3[i], "\n")
  cat("Specificity:", specificitiesRFF3[i], "\n")
  cat("kappa:",kappaRFF3[i],"\n")
  cat("\n")
}

#Accuracy: 1 
#Sensitivity: 1 
#Specificity: 1 
#kappa: 1
#The other models (Model 3, 4, 5, 6, 7, 8, 9, 10) all have perfect accuracy with a value of 1,

###### KNN WITH TRAIN #########

# build a structure to save data
accuraciesKNN3 <- vector("numeric", length = length(seeds))
confusion_matricesKNN3 <- list()
sensitivitiesKNN3 <- vector("numeric", length = length(seeds))
specificitiesKNN3 <- vector("numeric", length = length(seeds))
kappaKNN3 <- vector("numeric", length = length(seeds))

for (i in 1:length(seeds)) {
  set.seed(seeds[i])
  
  # test and train
  idx <- createDataPartition(Rdata3$disease, p = .7, list = FALSE, times = 1) 
  trn3 <- Rdata3[idx,]
  tst3 <- Rdata3[-idx,]
  
  #model
  knnFit3 <- train(disease ~ .,
                   data=trn3,
                   method = "knn",
                   trControl = train.control,
                   preProcess = c("center","scale"),
                   tuneLength = 20)
  # test
  pred_knn3 <- predict(knnFit3,tst3)
  cm<-confusionMatrix(pred_knn3,tst3$disease)
  
  accuracy <- cm$overall["Accuracy"]
  kappa<-cm$overall["Kappa"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  
  # save result
  accuraciesKNN3[i] <- accuracy
  confusion_matricesKNN3[[i]] <- cm$table
  sensitivitiesKNN3[i] <- sensitivity
  specificitiesKNN3[i] <- specificity
  kappaKNN3[i]<-kappa
}

#print model result
for (i in 1:length(seeds)) {
  cat("Modello", i, "\n")
  cat("Accuracy:", accuraciesKNN3[i], "\n")
  cat("Confusion Matrix:\n")
  print(confusion_matricesKNN3[[i]])
  cat("Sensitivity:", sensitivitiesKNN3[i], "\n")
  cat("Specificity:", specificitiesKNN3[i], "\n")
  cat("kappa:",kappaKNN3[i],"\n")
  cat("\n")
}

#Accuracy: 0.93/1
#Sensitivity: 0.771 
#Specificity: 0.88/1 
#kappa: 0.861
#I Modelli 2, 3, 6, 7, 10 hanno tutti un'accuratezza di 0.9333333, specificity varia tra 0.7777778 e 1
#note that models 5 have slightly lower specificity than the other models,
#which may indicate a greater tendency to misidentify negative observations.
### SVM WITH TRAIN ##########

## SVM radial kernel ##

# build a structure to save data
accuraciesSVM.R3 <- vector("numeric", length = length(seeds))
confusion_matricesSVM.R3 <- list()
sensitivitiesSVM.R3<- vector("numeric", length = length(seeds))
specificitiesSVM.R3 <- vector("numeric", length = length(seeds))
kappaSVM.R3 <- vector("numeric", length = length(seeds))

for (i in 1:length(seeds)) {
  set.seed(seeds[i])
  
  # test and train
  idx <- createDataPartition(Rdata3$disease, p = .7, list = FALSE, times = 1) 
  trn3 <- Rdata3[idx,]
  tst3 <- Rdata3[-idx,]
  
  #model
  svm.rad.3 <- train(disease~., data=trn3, method="svmRadial", trControl=train.control,
                     preProcess=c("center","scale"),
                     tuneGrid=expand.grid(C=c(0.2,0.5,0.8,1,2,3,4),
                                          sigma=c(0.0001,0.001,0.01,0.1,0.5)))
  
  pred.rad3<-predict(svm.rad.3,tst3)
  cm <-confusionMatrix(pred.rad3,tst3$disease)
  
  accuracy <- cm$overall["Accuracy"]
  kappa<-cm$overall["Kappa"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  
  # save result
  accuraciesSVM.R3[i] <- accuracy
  confusion_matricesSVM.R3[[i]] <- cm$table
  sensitivitiesSVM.R3[i] <- sensitivity
  specificitiesSVM.R3[i] <- specificity
  kappaSVM.R3[i]<-kappa
}

#print model result
for (i in 1:length(seeds)) {
  cat("Modello", i, "\n")
  cat("Accuracy:", accuraciesSVM.R3[i], "\n")
  cat("Confusion Matrix:\n")
  print(confusion_matricesSVM.R3[[i]])
  cat("Sensitivity:", sensitivitiesSVM.R3[i], "\n")
  cat("Specificity:", specificitiesSVM.R3[i], "\n")
  cat("kappa:",kappaSVM.R3[i],"\n")
  cat("\n")
}

#Accuracy:0.86 al 1.
#Sensitivity: 1 
#Specificity: 0.8888889 e 1 
#kappa:  0.7058824 e 1

## SVM polinomial kernel ##

# build a structure to save data
accuraciesSVM.P3 <- vector("numeric", length = length(seeds))
confusion_matricesSVM.P3 <- list()
sensitivitiesSVM.P3<- vector("numeric", length = length(seeds))
specificitiesSVM.P3 <- vector("numeric", length = length(seeds))
kappaSVM.P3 <- vector("numeric", length = length(seeds))

for (i in 1:length(seeds)) {
  set.seed(seeds[i])
  
  # test and train
  idx <- createDataPartition(Rdata3$disease, p = .7, list = FALSE, times = 1) 
  trn3 <- Rdata3[idx,]
  tst3 <- Rdata3[-idx,]
  
  #model
  svm.pol.3 <- train(disease~., data=trn3, method="svmPoly", trControl=train.control,
                     preProcess=c("center","scale"), 
                     tuneGrid=expand.grid(C=c(0.001,0.01, 0.1,1,10,100),
                                          degree=c(1,2,3,4), scale=1))
  pred.pol3<-predict(svm.pol.3,tst3)  
  cm <-confusionMatrix(pred.pol3,tst3$disease)
  
  accuracy <- cm$overall["Accuracy"]
  kappa<-cm$overall["Kappa"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  
  # save result
  accuraciesSVM.P3[i] <- accuracy
  confusion_matricesSVM.P3[[i]] <- cm$table
  sensitivitiesSVM.P3[i] <- sensitivity
  specificitiesSVM.P3[i] <- specificity
  kappaSVM.P3[i]<-kappa
}

#print model result
for (i in 1:length(seeds)) {
  cat("Modello", i, "\n")
  cat("Accuracy:", accuraciesSVM.P3[i], "\n")
  cat("Confusion Matrix:\n")
  print(confusion_matricesSVM.P3[[i]])
  cat("Sensitivity:", sensitivitiesSVM.P3[i], "\n")
  cat("Specificity:", specificitiesSVM.P3[i], "\n")
  cat("kappa:",kappaSVM.P3[i],"\n")
  cat("\n")
}

#Accuracy: variando dal 93,33% al 100%
#Sensitivity: 0.8333333 a 1 
#Specificity:  l'88,89% e il 100% 
#kappa:  86,49% al 100

### LOGISTIC MODEL ########

# build a structure to save data
accuraciesGLM3 <- vector("numeric", length = length(seeds))
confusion_matricesGLM3 <- list()
sensitivitiesGLM3<- vector("numeric", length = length(seeds))
specificitiesGLM3 <- vector("numeric", length = length(seeds))
kappaGLM3 <- vector("numeric", length = length(seeds))

for (i in 1:length(seeds)) {
  set.seed(seeds[i])
  
  # test and train
  idx <- createDataPartition(Rdata3$disease, p = .7, list = FALSE, times = 1) 
  trn3 <- Rdata3[idx,]
  tst3 <- Rdata3[-idx,]
  
  #model
  lr3 <- train(disease~., method="glm", family=binomial, data=trn3,
               trControl =train.control,preProcess = c("center","scale"))
  
  pred.lr3 <- predict(lr3, tst3)
  cm<-confusionMatrix(pred.lr3,tst3$disease, positive="1")
  
  #ROC plot
  pred.R1<-predict(lr3, tst3,type='prob')
  myroc7 <- roc(tst3$disease ~  pred.R1[,2],  plot=T, print.auc=T)
  
  accuracy <- cm$overall["Accuracy"]
  kappa<-cm$overall["Kappa"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  
  # save result
  accuraciesGLM3[i] <- accuracy
  confusion_matricesGLM3[[i]] <- cm$table
  sensitivitiesGLM3[i] <- sensitivity
  specificitiesGLM3[i] <- specificity
  kappaGLM3[i]<-kappa
}

#print model result
for (i in 1:length(seeds)) {
  cat("Modello", i, "\n")
  cat("Accuracy:", accuraciesGLM3[i], "\n")
  cat("Confusion Matrix:\n")
  print(confusion_matricesGLM3[[i]])
  cat("Sensitivity:", sensitivitiesGLM3[i], "\n")
  cat("Specificity:", specificitiesGLM3[i], "\n")
  cat("kappa:",kappaGLM3[i],"\n")
  cat("\n")
}

# Accuracy: da 0.8666667 a 1
# Specificity: da 0.8888889 a 1
#Kappa: da 0.7368421 a 1
# Sensibilità: da 0.6666667 a 1

library(pROC)
pred.R1<-predict(lr3, tst3,type='prob')
myroc7 <- roc(tst3$disease ~  pred.R1[,2],  plot=T)


### plot ROC graph ###

plot(myroc8, col = "blue", print.auc = F, main = "Curve ROC", xlab = "1 - Specificity", ylab = "Sensitivity")
lines(myroc48, col = "red")
lines(myroc13, col = "green")
legend("bottomright", legend = c("All features", "VIF within group ", "VIF between
groups and Lasso"), col = c("blue", "red", "green"), lty = 1)

