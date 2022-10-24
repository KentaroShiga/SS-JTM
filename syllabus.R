library(RMeCab)
library(dplyr)
library(magrittr)
library(tidyverse)
library(purrr)
library(stringr)
library(factoextra)
library(FactoMineR)
library('stringi')
library(zipangu)
library(ggdendro)
library(rgl)
library(tm)
library(topicmodels)
library(lda)
library(ldatuning)

K = 11
REC = 5 # 件数
# is_blank の定義
is_blank <- function(x) {is.na(x) | x == ""}

while(getwd() != "/Users/sugarcane") { setwd("../")}

# 前処理
data <- read.csv("R/full_scraping_result.csv", head = T, stringsAsFactor=T)
origin <- data
origin$学修内容 %<>% stri_trans_nfkc() # line 39
origin$学修内容 %<>% tolower() %<>% 
  stri_trans_nfkc() %<>% kansuji2arabic_all()

# 相談内容を全てtxtファイルとして出力
setwd("R/syllabusText")
for (i in 1:NROW(origin)) {
if(origin[i, ]$インデックス != 'E') {
  tmp <- paste0(origin[i, ]$授業科目名,
                origin[i, ]$授業の概要, 
                origin[i, ]$学修の目的,
                origin[i, ]$到達目標,
                origin[i, ]$学修内容) %>% as.character()
  tmp <-
    gsub("【", " ", tmp) 
  tmp <- 
    gsub(" 】", " ", tmp)
  tmp <- 
    gsub("・", " ", tmp)
  tmp <- 
    gsub("¥t", " ", tmp)
  tmp <- 
    gsub("\t", " ", tmp)
  writeLines(text = tmp, con = paste0(origin[i, ]$科目名, ".txt"))
}
}
# 単語文書行列の作成
prime <- docDF("./", type = 1, pos = c("動詞", "名詞", "形容詞", "副詞"))

# 単語の絞り込み
prime %<>% 
  filter(! TERM %in% 
           c("する", "なる", "いる", "れる", "t"))

# 「すべてが空欄である列」以外を残す（＝空列の除去）
 unnecessary_col <- apply(prime, 2,
                          function(x){
                           all(is_blank(x))                         })
prime <- prime[, !unnecessary_col]

# 列名を短縮化する
colnames(prime) %<>% str_replace(".txt", "")

# 重複するTERMの統合
prime1 <- prime %>% select(TERM, 4:NCOL(prime)) 
if(TRUE) {
for (i in 1:(NROW(prime1)-1)) {
  if(prime1[i,]$TERM == prime1[(i+1),]$TERM) {
    for (j in 2:(NCOL(prime1))) {
      prime1[(i+1), j] <- prime1[(i+1), j] + prime1[i, j]
    }
    prime1[i,]$TERM <- NA
  }
}
}
#prime1 <- group_by(prime1, TERM)
prime1 %<>% na.omit()
# 数値列だけ残したオブジェクトを作成
prime2 <- prime1 %>% select(-c(TERM))
rownames(prime2) <- prime1$TERM
prime2a <- prime2  %>% t() %>% as.DocumentTermMatrix(weighting = weightTf)
raw.sum <- apply(prime2a, 1, FUN = sum)
prime2a <- prime2a[raw.sum!=0,]

if(TRUE) {
findK <- FindTopicsNumber(
  prime2a,
  topics = seq(from=5, to=20, by=1),
  metrics=c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
)
}
res1 <- prime2a %>% LDA(K)
result <- posterior(res1)[[2]]
prime3 <- dtm2ldaformat(prime2a)
res2 <- lda.collapsed.gibbs.sampler(prime3$documents, K = K, 
                                    prime3$vocab, 25, 0.1, 0.1, 
                                    compute.log.likelihood = TRUE)

top.topic.words(res2$topics, 10, by.score = TRUE)

findK %>% head()
FindTopicsNumber_plot(findK)

