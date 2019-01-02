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

#Process text and create corpus
library(dplyr)
library(tm)
library(stringr)
wine_corpus = Corpus(VectorSource(wine$description))
wine_corpus = tm_map(wine_corpus, removePunctuation)
wine_corpus = tm_map(wine_corpus, content_transformer(tolower))
wine_corpus = tm_map(wine_corpus, removeNumbers)
wine_corpus = tm_map(wine_corpus, removeWords, c("the", "and", stopwords("english")))
wine_corpus = tm_map(wine_corpus, stripWhitespace)
wine_corpus <- tm_map(wine_corpus, stemDocument)

wine_dtm_tfidf <- DocumentTermMatrix(wine_corpus, control = list(weighting = weightTfIdf))
#wine_dtm_tfidf <- DocumentTermMatrix(wine_corpus)
wine_dtm_tfidf <- removeSparseTerms(wine_dtm_tfidf, 0.95)
wine_dtm_tfidf <- as.matrix(wine_dtm_tfidf)


# split train/test
n = dim(wine)[1]
set.seed(12345)
# id = sample(1:n, floor(n*0.8))
# train = wine[id,]
# test = wine[-id,]

#small for test model
id2 = sample(1:n, floor(n*0.1))
wine_sample <- wine_dtm_tfidf[id2,]
wine2 <- wine[id2,]
n2 = length(id2)
id_test = sample(1:n2, floor(n2*0.8))
train = wine_sample[id_test,]
test = wine_sample[-id_test,]


#create the train set
classsify <- wine2[id_test, 15]
wine_training_set <- cbind(train, classsify)

#create the test result
wine_testing_result <- wine2[-id_test, 15]




library(caret)
ctrl <- trainControl(method="repeatedcv",   # 10fold cross validation
                     repeats=5,		    # do 5 repititions of cv
                     summaryFunction=twoClassSummary,	# Use AUC to pick the best model
                     classProbs=TRUE)

train_model <- train(x=train,
                  y= classsify,
                  method = "svmRadial",   # Radial kernel
                  tuneLength = 9,					# 9 values of the cost function
                  preProc = c("center","scale"),  # Center and scale data
                  metric="ROC",
                  trControl=ctrl)


##train old
#colnames(wine_training_set)[ncol(wine_training_set)] <- "y"
#wine_training_set <- as.data.frame(wine_training_set)
#wine_training_set$y <- as.factor(wine_training_set$y)
#train_model <- train(y ~., data = wine_training_set, method = 'svmLinear3')
#train_model <- train(y ~., data = wine_training_set, method = 'naive_bayes')
#train_model <- train(y ~., data = wine_training_set, method = 'svmRadial')


#Build the prediction  
model_result <- predict(train_model, newdata = test)

conf_train <- table(model_result, wine_testing_result)
names(dimnames(conf_train)) <- c("Predicted class", "Actual class")
confusionMatrix(conf_train)

