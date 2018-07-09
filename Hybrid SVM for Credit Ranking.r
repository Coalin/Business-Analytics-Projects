library(e1071)
library(randomForest)
library(pROC)
library(ggplot2)
library(caret)

setwd("/Users/Colin/Desktop/Teamwork")
Attr = c("Existing_checking_account",
         "Duration",                     
         "Credit_history",               
         "Purpose",                      
         "Credit_amount",                
         "Savings_account",              
         "Present_employment",           
         "Installment_rate",             
         "Personal_status",              
         "Other_debtors",                
         "Present_residence",            
         "Property",                     
         "Age",                          
         "Other_installment_plans",      
         "Housing",                      
         "Existing_credits",             
         "Job",                          
         "Liable_to_provide_maintenance",
         "Telephone",                    
         "Foreign_worker",               
         "Credit")

data = read.csv(file = "GermanCreditData.csv", 
                header = TRUE, 
                col.names = Attr
)
data$Credit = as.factor(data$Credit)

#resampling for a balance
balance <- function(data,yval) {
  y.vector <- with(data,get(yval))
  index.0 <- which(y.vector==1)
  index.1 <- which(y.vector==2)
  index.1 <- sample(index.1, length(index.0), replace = TRUE)
  result <- data[sample(c(index.0,index.1)),]
  result
}
data <- balance(data, "Credit")

set.seed(12345)
train_RowIDs = sample(1:nrow(data), nrow(data)*0.7)
train_Data = data[train_RowIDs,]
test_Data = data[-train_RowIDs,]

table(data$Credit)
table(train_Data$Credit)
table(test_Data$Credit)

model = svm(Credit ~ .,
            data = train_Data,
            type = "C-classification", 
            kernel = "radial", 
            #            kernel = "linear",
            cost = 10, gamma = 0.1) 
summary(model)
ind_train_data = train_Data[,1:20]
pred_Train = predict(model, ind_train_data) 
cm_Train = table(train_Data$Credit, pred_Train)
print(cm_Train)

ind_test_data = test_Data[,1:20]
pred_Test = predict(model, ind_test_data) 
cm_Test = table(test_Data$Credit, pred_Test)
print(cm_Test)
accu_Test= sum(diag(cm_Test))/sum(cm_Test)

rf = randomForest(Credit ~ ., data = train_Data, importance = TRUE)
DrawL<-par()
par(mfrow=c(2,1),mar=c(5,5,3,1))
plot(rf,main="Random Forest Error VS Number of Decision Trees")
plot(margin(rf),type="h",main="Margin Detection",
     xlab="Observations",ylab="Ration") 

rf1 = randomForest(Credit ~ ., data = train_Data, importance = TRUE, ntree = 140)
importance(rf1,type=1)
varImpPlot(x=rf1, sort=TRUE, n.var=nrow(rf$importance),main="Importance of Variables")

rm(list = ls())
setwd("/Users/Colin/Desktop/Teamwork")
Attr = c("Existing_checking_account",
         "Duration",                     
         "Credit_history",               
         "Purpose",                      
         "Credit_amount",                
         "Savings_account",              
         "Present_employment",           
         "Installment_rate",             
         "Personal_status",              
         "Other_debtors",                
         "Present_residence",            
         "Property",                     
         "Age",                          
         "Other_installment_plans",      
         "Existing_credits",             
         "Job",                          
         "Telephone",                    
         "Credit")

data = read.csv(file = "Credit.csv", 
                header = TRUE, 
                col.names = Attr
)
data$Credit = as.factor(data$Credit)


#resampling for a balance
balance <- function(data,yval) {
  y.vector <- with(data,get(yval))
  index.0 <- which(y.vector==1)
  index.1 <- which(y.vector==2)
  index.1 <- sample(index.1, length(index.0), replace = TRUE)
  result <- data[sample(c(index.0,index.1)),]
  result
}
data <- balance(data, "Credit")

set.seed(12345)
train_RowIDs = sample(1:nrow(data), nrow(data)*0.7)
train_Data = data[train_RowIDs,]
test_Data = data[-train_RowIDs,]

table(data$Credit)
table(train_Data$Credit)
table(test_Data$Credit)

model = svm(Credit ~ .,
            data = train_Data,
            type = "C-classification", 
            kernel = "radial", 
            #            kernel = "linear",
            cost = 100, gamma = 0.01) 
summary(model)
ind_train_data = train_Data[,1:17]
pred_Train = predict(model, ind_train_data) 
cm_Train = table(train_Data$Credit, pred_Train)
print(cm_Train)

ind_test_data = test_Data[,1:17]
pred_Test = predict(model, ind_test_data) 
cm_Test = table(test_Data$Credit, pred_Test)
print(cm_Test)
accu_Test= sum(diag(cm_Test))/sum(cm_Test)