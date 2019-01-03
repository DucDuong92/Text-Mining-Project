#Load and preprocess data
wine <- read.csv("C:/Users/Duong Minh Duc/Documents/GitHub/Text-Mining-Project/wine.csv")
wine <- na.omit(wine)
wine$quality <- wine$points > 90
wine$quality[wine$quality == TRUE] <- "excellent"
wine$quality[wine$quality == FALSE] <- "good"
wine$quality <- as.factor(wine$quality)
wine$value <- wine$points/log(wine$price)
wine$benefit <- wine$value > 30
wine$benefit[wine$benefit == TRUE] <- "high"
wine$benefit[wine$benefit == FALSE] <- "medium"
wine$benefit <- as.factor(wine$benefit)
wine$description <- paste(wine$description, wine$country, wine$designation, wine$province, wine$region_1, wine$region_2, wine$variety)
wine$description <- as.character(wine$description)

# split train/test
n = dim(wine)[1]
set.seed(12345)
# id = sample(1:n, floor(n*0.8))
# train = wine[id,]
# test = wine[-id,]

#small for test model
id2 = sample(1:n, floor(n*0.2))
wine_sample <- wine[id2,]
#wine2 <- wine[id2,]
n2 = length(id2)
id_test = sample(1:n2, floor(n2*0.8))
train = wine_sample[id_test,]
test = wine_sample[-id_test,]


#Process text and create corpus
library(dplyr)
library(tm)
library(stringr)

##clean function
clean <- function(text_vector)
  {
    wine_corpus = Corpus(VectorSource(text_vector))
    wine_corpus = tm_map(wine_corpus, removePunctuation)
    wine_corpus = tm_map(wine_corpus, content_transformer(tolower))
    wine_corpus = tm_map(wine_corpus, removeNumbers)
    wine_corpus = tm_map(wine_corpus, removeWords, c("the", "and", stopwords("english")))
    wine_corpus = tm_map(wine_corpus, stripWhitespace)
    wine_corpus <- tm_map(wine_corpus, stemDocument)
    
    return(wine_corpus)
  }

##create the train set
wine_train_set <- clean(train$description)
train_dtm_tfidf <- DocumentTermMatrix(wine_train_set, control = list(weighting = weightTfIdf))
#train_dtm_tfidf <- DocumentTermMatrix(wine_train_set)
train_dtm_tfidf <- removeSparseTerms(train_dtm_tfidf, 0.95)

#wine_train_set <- cbind(wine_train_set, train$quality)

#create the test set
wine_test_set <- clean(test$description)
wine_test_set <- DocumentTermMatrix(wine_test_set, control = list(dictionary = Terms(train_dtm_tfidf) ,weighting = weightTfIdf))
#wine_test_set <- DocumentTermMatrix(wine_test_set, control = list(dictionary = Terms(train_dtm_tfidf)))

#create matrix for training
wine_train_set <- as.matrix(train_dtm_tfidf)
wine_test_set <- as.matrix(wine_test_set)
wine_test_set <- wine_test_set[,Terms(train_dtm_tfidf)]
#create the test result
wine_testing_result <- test$quality


# Trainning model
library(caret)
# ctrl <- trainControl(method="repeatedcv",   # 10fold cross validation
#                      repeats=5,		    # do 5 repititions of cv
#                      summaryFunction=twoClassSummary,	# Use AUC to pick the best model
#                      classProbs=TRUE)
# 
# train_model <- train(x=wine_train_set,
#                   y= train$quality,
#                   #method = "svmRadial",   # Radial kernel
#                   method = "naive_bayes",
#                   tuneLength = 9,					# 9 values of the cost function
#                   preProc = c("center","scale"),  # Center and scale data
#                   metric="ROC",
#                   trControl=ctrl)


##train old
#colnames(wine_train_set)[ncol(wine_train_set)] <- "y"
#wine_train_set <- as.data.frame(wine_train_set)
#wine_train_set$y <- as.factor(wine_train_set$y)
#train_model <- train(y ~., data = wine_training_set, method = 'svmLinear3')
train_model <- train(x= wine_train_set, y=train$quality , method = 'naive_bayes')
#train_model <- train(y ~., data = wine_training_set, method = 'svmRadial')
#train_model <- train(y ~., data = wine_training_set, method = 'gbm')
#train_model <- train(y ~., data = wine_training_set, method = 'nnet')


#Build the prediction  
model_result <- predict(train_model, newdata = wine_test_set)

conf_train <- table(model_result, wine_testing_result)
names(dimnames(conf_train)) <- c("Predicted class", "Actual class")
confusionMatrix(conf_train)

# check_accuracy <- as.data.frame(cbind(prediction = model_result,  classify = wine_testing_result))
# 
# check_accuracy <- check_accuracy %>% mutate(prediction = as.integer(prediction))
# 
# check_accuracy$accuracy <- if_else(check_accuracy$prediction == check_accuracy$classify, 1, 0)
# round(prop.table(table(check_accuracy$accuracy)), 3)
