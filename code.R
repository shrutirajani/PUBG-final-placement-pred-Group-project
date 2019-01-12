---
title: "R Notebook"
---

```{r}
library(corrplot)
library(gplots)
library(FNN)
library(caret)
library(dummies)
train <- read.csv('train_V2.csv',stringsAsFactors = F,nrows = 500000)
test <- read.csv('test_V2.csv',stringsAsFactors = F,nrows = 500000)
test$winPlacePerc = NA

str(train)
summary(train)
sapply(train,class)
```
```{r}
# Remove IDs
train <- train[,-c(1:3)]
train$matchType <- as.factor(train$matchType)
```




```{r}
for(i in 1:4) {
  hist(train[,i], main=names(train)[i])
}

for(i in 5:8) {
  hist(train[,i], main=names(train)[i])
}

for(i in 9:12) {
  hist(train[,i], main=names(train)[i])
}

for(i in 14:17) {
  hist(train[,i], main=names(train)[i])
}

for(i in 18:22) {
  hist(train[,i], main=names(train)[i])
}

for(i in 22:26) {
  hist(train[,i], main=names(train)[i])
}
```

```{r}
for(i in 1:4) {
  boxplot(train[,i], main=names(train)[i])
}

for(i in 5:8) {
  boxplot(train[,i], main=names(train)[i])
}

for(i in 9:12) {
  boxplot(train[,i], main=names(train)[i])
}

for(i in 14:17) {
  boxplot(train[,i], main=names(train)[i])
}

for(i in 18:22) {
  boxplot(train[,i], main=names(train)[i])
}

for(i in 22:26) {
  boxplot(train[,i], main=names(train)[i])
}




```

```{r}
correlations = cor(train[,-c(13)])
corrplot(correlations, method="square")

```







```{r}
# Split into train and val
set.seed(999)
num_rows <- nrow(train)
train_idx <- sample(num_rows,floor(0.80*num_rows))

new_train <- train[train_idx,]
val <- train[-train_idx,]

```


```{r}
# Fit Linear Reg Model
linear.reg <- lm(winPlacePerc ~.,data = new_train)


summary(linear.reg)
```

```{r}
# Predict on validation set
preds <- predict(linear.reg, val)
RMSE(preds,val$winPlacePerc)
```

```{r}
# Normalize
norm.values <- preProcess(new_train, method=c("center", "scale"))
new_train.norm <- predict(norm.values, new_train)
val.norm <- predict(norm.values, val)

# Dummy variables for matchType because it's factor
new_train.norm <- dummy.data.frame(new_train.norm)
val.norm <- dummy.data.frame(val.norm)


# KNN Model Trial with k = 3
knn_model <-knn(train = new_train.norm,test = val.norm,cl = new_train.norm$winPlacePerc,k = 3)
knn_preds <- as.numeric(as.character(knn_model))
RMSE(val.norm$winPlacePerc,knn_preds)


```
```{r}
# Find K based on RMSE 
accuracy.df <- data.frame(k = seq(1, 14, 1), accuracy = rep(0, 14))

### compute knn for different k on validation

for(i in 1:14) {
  knn.pred <-knn(train = new_train.norm,test = val.norm,cl = new_train.norm$winPlacePerc,k = i)
  knn.preds_num <- as.numeric(as.character(knn.pred))
  accuracy.df[i, 2] <- RMSE(val.norm$winPlacePerc,knn.preds_num)
  print(accuracy.df[i, 2])
}
write.csv(accuracy.df,'knn_acc.csv')
```

```{r}
trainControl <- trainControl(method="cv",
                             number=5,
                             verbose = TRUE)

#RSME taken as metric 
metric <- "RMSE"

# CART Model
set.seed(999)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(winPlacePerc~.,
                  data=new_train, 
                  method="rpart",
                  metric=metric,
                  tuneGrid=grid,
                  preProc=c("center","scale"),
                  trControl=trainControl)







```






