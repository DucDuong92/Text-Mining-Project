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
wine$description <- as.character(wine$description)



# split train/test
n = dim(wine)[1]
set.seed(12345)
id = sample(1:n, floor(n*0.8))
train = wine[id,]
test = wine[-id,]

#small for test model
id2 = sample(1:n, floor(n*0.1))
wine2 <- wine[id2,]
n2 = length(id2)
id_test = sample(1:n2, floor(n2*0.8))
train = wine2[id_test,]
test = wine2[-id_test,]



#Process texts
library(dplyr)
#library(text2vec)
#library(SnowballC)
library(tm)
library(stringr)

clean_corpus <- function(input)
  {
    wine_corpus = Corpus(VectorSource(input))
    wine_corpus = tm_map(wine_corpus, removePunctuation)
    wine_corpus = tm_map(wine_corpus, content_transformer(tolower))
    wine_corpus = tm_map(wine_corpus, removeNumbers)
    wine_corpus = tm_map(wine_corpus, removeWords, c("the", "and", stopwords("english")))
    wine_corpus = tm_map(wine_corpus, stripWhitespace)
    wine_corpus <- tm_map(wine_corpus, stemDocument)
    
    #wine_dtm_tfidf <- DocumentTermMatrix(wine_corpus, control = list(weighting = weightTfIdf))
    wine_dtm_tfidf <- DocumentTermMatrix(wine_corpus)
    wine_dtm_tfidf = removeSparseTerms(wine_dtm_tfidf, 0.95)
    wine_dtm_tfidf <- as.matrix(wine_dtm_tfidf)
    
    return(wine_dtm_tfidf)
  }

#create the train
wine_dtm_tfidf <- clean_corpus(train$description)
wine_training_set <- cbind(wine_dtm_tfidf, train$benefit)

#create the test
test_dtm <- clean_corpus(test$description)

##
colnames(wine_training_set)[ncol(wine_training_set)] <- "y"

wine_training_set <- as.data.frame(wine_training_set)
wine_training_set$y <- as.factor(wine_training_set$y)

library(caret)
#train_model <- train(y ~., data = wine_training_set, method = 'svmLinear3')
train_model <- train(y ~., data = wine_training_set, method = 'naive_bayes')
#train_model <- train(y ~., data = wine_training_set, method = 'svmRadial')


#Build the prediction  
model_result <- predict(train_model, newdata = test_dtm)

check_accuracy <- as.data.frame(cbind(prediction = model_result,  quality = test$benefit))

check_accuracy <- check_accuracy %>% mutate(prediction = as.integer(prediction) )

check_accuracy$accuracy <- if_else(check_accuracy$prediction == check_accuracy$quality, 1, 0)
round(prop.table(table(check_accuracy$accuracy)), 3)

