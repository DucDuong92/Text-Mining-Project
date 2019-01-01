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


#Process texts
library(dplyr)
library(text2vec)
library(SnowballC)
library(tm)
library(stringr)
wine_corpus = Corpus(VectorSource(train$description))
wine_corpus = tm_map(wine_corpus, removePunctuation)
wine_corpus = tm_map(wine_corpus, content_transformer(tolower))
wine_corpus = tm_map(wine_corpus, removeNumbers)
wine_corpus = tm_map(wine_corpus, removeWords, c("the", "and", stopwords("english")))
wine_corpus = tm_map(wine_corpus, stripWhitespace)
wine_corpus <- tm_map(wine_corpus, stemDocument)

wine_dtm_tfidf <- DocumentTermMatrix(wine_corpus, control = list(weighting = weightTfIdf))
#wine_dtm_tfidf = removeSparseTerms(wine_dtm_tfidf, 0.95)
wine_dtm_tfidf

wine_train_vector <- as.matrix(wine_dtm_tfidf)
wine_training_set <- cbind(wine_train_vector, train$quality)


# Old code
# wine$description <- gsub('[[:punct:] ]+',' ',wine$description)
# wine$description<- gsub('[[:digit:]]+', '', wine$description)
# wine$description <- tolower(wine$description)
# 
# stopwords_regex = paste(stopwords('en'), collapse = '\\b|\\b')
# stopwords_regex = paste0('\\b', stopwords_regex, '\\b')
# wine$description = stringr::str_replace_all(wine$description, stopwords_regex, '')
# 
# 
# wine$description <- str_split(wine$description, ' ')

