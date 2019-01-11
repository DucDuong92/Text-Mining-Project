#Load and preprocess data
wine <- read.csv("C:/Users/Duong Minh Duc/Documents/GitHub/Text-Mining-Project/wine.csv")
wine <- na.omit(wine)
wine$quality <- wine$points > 88
wine$quality[wine$quality == TRUE] <- "excellent"
wine$quality[wine$quality == FALSE] <- "good"
wine$quality <- as.factor(wine$quality)
wine$value <- wine$points/log(wine$price)
wine$benefit <- wine$value > 27
wine$benefit[wine$benefit == TRUE] <- "high"
wine$benefit[wine$benefit == FALSE] <- "medium"
wine$benefit <- as.factor(wine$benefit)
wine$description <- paste(wine$description, wine$country, wine$designation, wine$province, wine$region_1, wine$region_2, wine$variety)
wine$description <- as.character(wine$description)

#Convert Wine type language
#Replace german names with English names for wines

wine$description <- gsub("weissburgunder", "chardonnay", wine$description)
wine$description <- gsub("spatburgunder", "pinot noir", wine$description)
wine$description <- gsub("grauburgunder", "pinot gris", wine$description)

#Replace the Spanish garnacha with the french grenache
wine$description <- gsub("garnacha", "grenache", wine$description)

#Replace the Italian pinot nero with the french pinot noir
wine$description <- gsub("pinot nero", "pinot noir", wine$description)

#Replace the Portugues alvarinho with the spanish albarino
wine$description <- gsub("alvarinho", "albarino", wine$description)

#wine$description <- wine[which(!grepl("[^\x01-\x7F]+", wine$description)),]

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
library(NLP)
library(openNLP)

#update stop words:

stopwords <- stopwords("english")
stopwords <- stopwords[!stopwords=="very"]
stopwords <- c("the", "and", "wine", stopwords)

##clean function
clean <- function(text_vector)
  {
    wine_corpus = VCorpus(VectorSource(text_vector))
    wine_corpus = tm_map(wine_corpus, removePunctuation)
    wine_corpus = tm_map(wine_corpus, content_transformer(tolower))
    wine_corpus = tm_map(wine_corpus, removeNumbers)
    wine_corpus = tm_map(wine_corpus, removeWords, stopwords )
    #wine_corpus = tm_map(wine_corpus, stripWhitespace)
    wine_corpus <- tm_map(wine_corpus, stemDocument)
    
    
    return(wine_corpus)
  }

NLP_tokenizer <- function(x) {
  unlist(lapply(ngrams(words(x), 1:1), paste, collapse = "_"), use.names = FALSE)
}

##create the train set
wine_train_set <- clean(train$description)


#corpus <- Corpus(VectorSource(texts))
#matrix <- DocumentTermMatrix(corpus,control=list(tokenize=tokenize_ngrams))

train_dtm_tfidf <- DocumentTermMatrix(wine_train_set, control = list(weighting = weightTfIdf, tokenize=NLP_tokenizer))
#train_dtm_tfidf <- DocumentTermMatrix(wine_train_set, control = list( tokenize=NLP_tokenizer))
#train_dtm_tfidf <- DocumentTermMatrix(wine_train_set)
train_dtm_tfidf <- removeSparseTerms(train_dtm_tfidf, 0.99)

#wine_train_set <- cbind(wine_train_set, train$quality)

#create the test set
wine_test_set <- clean(test$description)
wine_test_set <- DocumentTermMatrix(wine_test_set, control = list(dictionary = Terms(train_dtm_tfidf) ,weighting = weightTfIdf, tokenize=NLP_tokenizer))
#wine_test_set <- DocumentTermMatrix(wine_test_set, control = list(dictionary = Terms(train_dtm_tfidf) , tokenize=NLP_tokenizer))

#create matrix for training
wine_train_set <- as.matrix(train_dtm_tfidf)
wine_test_set <- as.matrix(wine_test_set)
wine_test_set <- wine_test_set[,Terms(train_dtm_tfidf)]
#create the test result
wine_testing_result <- test$quality





# #tagPos
# 
# tagPOS <-  function(x, ...) {
#   s <- as.String(x)
#   word_token_annotator <- Maxent_Word_Token_Annotator()
#   a2 <- Annotation(1L, "sentence", 1L, nchar(s))
#   a2 <- annotate(s, word_token_annotator, a2)
#   a3 <- annotate(s, Maxent_POS_Tag_Annotator(), a2)
#   a3w <- a3[a3$type == "word"]
#   POStags <- unlist(lapply(a3w$features, `[[`, "POS"))
#   POStagged <- paste(sprintf("%s/%s", s[a3w], POStags), collapse = " ")
#   list(POStagged = POStagged, POStags = POStags)
# }
# 
# #Weight for nouns:
# tag <- tagPOS(Terms(train_dtm_tfidf))
# tag <- tag$POStags
# noun_id <- which( tag=="NN")
# nouns <- colnames(wine_train_set)[noun_id]

# #multify for noun
# for (i in 1:dim(wine_train_set)[2]) {
#   if (colnames(wine_train_set)[i] %in% nouns ) {
#     wine_train_set[,i] <- wine_train_set[,i]*2
#     #wine_test_set[,i] <- wine_test_set[,i]*2
#   }
# }


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
#train_model <- train(x= wine_train_set, y=train$quality , method = 'svmLinear3')
train_nb_model <- train(x= wine_train_set, y=train$quality , method = 'naive_bayes')
#train_model <- train(x= wine_train_set, y=train$quality , method = 'svmRadial')
#train_model <- train(x= wine_train_set, y=train$quality , method = 'gbm')
#train_model <- train(x= wine_train_set, y=train$quality , method = 'nnet')


#Build the prediction  
model_result <- predict(train_nb_model, newdata = wine_test_set)

conf_train <- table(model_result, wine_testing_result)
names(dimnames(conf_train)) <- c("Predicted class", "Actual class")
confusionMatrix(conf_train)
# check_accuracy <- as.data.frame(cbind(prediction = model_result,  classify = wine_testing_result))
# 
# check_accuracy <- check_accuracy %>% mutate(prediction = as.integer(prediction))
# 
# check_accuracy$accuracy <- if_else(check_accuracy$prediction == check_accuracy$classify, 1, 0)
# round(prop.table(table(check_accuracy$accuracy)), 3)
