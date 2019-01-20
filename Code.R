#Load and preprocess data
wine <- read.csv("C:/Users/Duong Minh Duc/Documents/GitHub/Text-Mining-Project/wine.csv")
wine <- na.omit(wine)
wine[duplicated(wine),]
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

#clean function
wine$description <- iconv(wine$description, from = "UTF-8", to = "ASCII", sub = "")

# split train/test
n = dim(wine)[1]
set.seed(12345)

#small for test model
id2 = sample(1:n, floor(n*0.2))
wine_sample <- wine[id2,]
wine2 <- wine[id2,]
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

##create the train set
wine_train_set <- clean(train$description)

NLP_tokenizer <- function(x) {
  unlist(lapply(ngrams(words(x), 1:2), paste, collapse = "_"), use.names = FALSE)
}

train_dtm_tfidf <- DocumentTermMatrix(wine_train_set, control = list(weighting = weightTfIdf, tokenize=NLP_tokenizer))
#train_dtm_tfidf <- DocumentTermMatrix(wine_train_set, control = list( tokenize=NLP_tokenizer))
#train_dtm_tfidf <- DocumentTermMatrix(wine_train_set)
train_dtm_tfidf <- removeSparseTerms(train_dtm_tfidf, 0.99)




#create the test set
wine_test_set <- clean(test$description)
wine_test_set <- DocumentTermMatrix(wine_test_set, control = list(dictionary = Terms(train_dtm_tfidf) ,weighting = weightTfIdf, tokenize=NLP_tokenizer))
#wine_test_set <- DocumentTermMatrix(wine_test_set, control = list(dictionary = Terms(train_dtm_tfidf) , tokenize=NLP_tokenizer))

#create matrix for training
wine_train_set <<- as.matrix(train_dtm_tfidf)
wine_test_set <- as.matrix(wine_test_set)
wine_test_set <- wine_test_set[,Terms(train_dtm_tfidf)]
#create the test result
wine_testing_result <- test$benefit


# #tagPos
# tagPOS <-  function(x, ...) {
#   s <- as.String(x)
#   word_token_annotator <- Maxent_Word_Token_Annotator()
#   a2 <- Annotation(1L, "sentence", 1L, nchar(s))
#   a2 <- NLP::annotate(s, word_token_annotator, a2)
#   a3 <- NLP::annotate(s, Maxent_POS_Tag_Annotator(), a2)
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
# adj_id <- which( tag=="JJ")
# adj <- colnames(wine_train_set)[adj_id]
# 
# column_id <- c()
# 
# #multify for noun
# column_id <- c()
# for (i in 1:dim(wine_train_set)[2]) {
#   check <- colnames(wine_train_set)[i] %in% adj
#   if(check)
#     {
#       column_id <- c(column_id, i)
#     }
# }
# 
# wine_train_set[,column_id] <- wine_train_set[,column_id]*2
# wine_test_set[,column_id] <- wine_test_set[,column_id]*2

# Trainning model
library(caret)

##train old
#colnames(wine_train_set)[ncol(wine_train_set)] <- "y"
#wine_train_set <- as.data.frame(wine_train_set)
#wine_train_set$y <- as.factor(wine_train_set$y)
train_svm_model <- train(x= wine_train_set, y=train$benefit , method = 'svmLinear3')
#train_nb_model <- train(x= wine_train_set, y=train$benefit , method = 'naive_bayes')
#train_svmRBF_model <- train(x= wine_train_set, y=train$qual , method = 'svmRadial')
#train_model <- train(x= wine_train_set, y=train$quality , method = 'gbm')
#train_model <- train(x= wine_train_set, y=train$quality , method = 'nnet')


#Build the prediction  
model_result <- predict(train_svm_model, newdata = wine_test_set)

conf_train <- table(model_result, wine_testing_result)
names(dimnames(conf_train)) <- c("Predicted class", "Actual class")
confusionMatrix(conf_train)


#top influence
#varImp(train_svm_model)

#top tf-idf
good<- which(train$quality=="good")
good <- wine_train_set[good,]
excellent<- which(train$quality=="excellent")
excellent <- wine_train_set[excellent,]
high<- which(train$benefit=="high")
high <- wine_train_set[high,]
medium<- which(train$benefit=="medium")
medium <- wine_train_set[medium,]

good = data.frame(sort(colSums(good), decreasing=TRUE))
wordcloud(rownames(good), good[,1], max.words=100, colors=brewer.pal(8, "Dark2"), scale=c(4,.5))

excellent = data.frame(sort(colSums(excellent), decreasing=TRUE))
wordcloud(rownames(excellent), excellent[,1], max.words=100, colors=brewer.pal(8, "Dark2"), scale=c(4,.5))

high = data.frame(sort(colSums(high), decreasing=TRUE))
wordcloud(rownames(high), high[,1], max.words=100, colors=brewer.pal(8, "Dark2"), scale=c(4,.5))

medium = data.frame(sort(colSums(medium), decreasing=TRUE))
wordcloud(rownames(medium), medium[,1], max.words=100, colors=brewer.pal(8, "Dark2"), scale=c(4,.5))


ggplot(subset(wine, price <= 100),
       aes(x = price, y = points)) +
  geom_point(alpha = 0.3,  position = position_jitter()) + 
  stat_smooth(method = "lm", size =2) +
  labs(title = 'Price vs Point for Wines $100 and Under') +
  theme_bw()

