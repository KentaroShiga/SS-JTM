import random
import sys
import numpy as np
from numpy.lib.index_tricks import ndindex
from scipy.sparse import lil_matrix
import MeCab 
import unicodedata
#from kanjize import int2kanji, kanji2int
import pandas as pd
import glob
import math
import matplotlib.pyplot as plt
import time
import datetime
import pytz
import os

#結合トピックモデルの本体となるこの部分はHatena Blogから引用
#https://yamaguchiyuto.hatenablog.com/entry/2017/03/22/100000


class JTM:
    def __init__(self, K, alpha, beta, max_iter, verbose=0):
        self.K=K
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.verbose=verbose

    def fit(self,X,V):
        self._X = X
        self._T = len(X) # number of vocab types
        self._N = len(X[0]) # number of documents
        self._V = V # number of vocabularies for each t

        self.Z = self._init_Z()
        self.ndk, self.nkv = self._init_params()
        self.nk = {}
        self.nd = {}
        for t in range(self._T):
            self.nk[t] = self.nkv[t].sum(axis=1)
            self.nd[t] = self.ndk[t].sum(axis=1)
        remained_iter = self.max_iter
        while True:
            if self.verbose: print(remained_iter)
            for t in np.random.choice(self._T, self._T, replace=False):
                for d in np.random.choice(self._N, self._N, replace=False):
                    for i in np.random.choice(len(self._X[t][d]), len(self._X[t][d]), replace=False):
                        k = self.Z[t][d][i]
                        v = self._X[t][d][i]

                        self.ndk[t][d][k] -= 1
                        self.nkv[t][k][v] -= 1
                        self.nk[t][k] -= 1
                        self.nd[t][d] -= 1

                        self.Z[t][d][i] = self._sample_z(t,d,v,self.nk[t])

                        self.ndk[t][d][self.Z[t][d][i]] += 1
                        self.nkv[t][self.Z[t][d][i]][v] += 1
                        self.nk[t][self.Z[t][d][i]] += 1
                        self.nd[t][d] += 1
            remained_iter -= 1
            if remained_iter <= 0: break
        return self

    def _init_Z(self):
        Z = {}
        for t in range(self._T):
            Z[t] = []
            for d in range(len(self._X[t])):
              Z[t].append(np.random.randint(low=0,high=self.K,size=len(self._X[t][d])))
        return Z

    def _init_params(self):
        ndk = {}
        nkv = {}
        for t in range(self._T):
            ndk[t] = np.zeros((self._N,self.K)) + self.alpha
            nkv[t] = np.zeros((self.K,self._V[t])) + self.beta
            for d in range(self._N):
                for i in range(len(self._X[t][d])):
                    k = self.Z[t][d][i]
                    v = self._X[t][d][i]
                    ndk[t][d,k]+=1
                    nkv[t][k,v]+=1
        return ndk,nkv

    def _sample_z(self,t,d,v,nk): # 周辺化ギブスサンプリング
        nkv = self.nkv[t][:,v] # k-dimensional vector

        prob = (sum([self.ndk[t][d] for t in range(self._T)])-self.alpha*(self._T-1)) *  (nkv/nk)
        prob = prob/prob.sum()
        z = np.random.multinomial(n=1, pvals=prob).argmax()
        return z

#dicdir = '-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-unidic-neologd'
#tagger2 = MeCab.Tagger(dicdir)
tagger = MeCab.Tagger('-Owakati')
tagger2 = MeCab.Tagger('')
select_conditions = ['動詞', '形容詞', '名詞']
stopword_list = ['ある', 'いる', 'なる', 'する', 'よう', 'こと', 'の'
                 'もの', 'れる', 'られる', '的', '回', 'ため', 'いく',
                 '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
                 '11', '12', '13', '14', '15', '16', 
                 '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', 
                 '十一', '十二', '十三', '十四', '十五', '十六']
normalize_list = ["動詞", "形容詞"]

# 前処理関数の実装
def preprocess(texts):
  # ディクショナリを初期化
  word_to_id = {}
  id_to_word = {}
  corpus = []
  for i in range(len(texts)):
    # 前処理
    text = str(texts[i])
    text = text.lower() # 小文字に変換
    text = unicodedata.normalize("NFKC", text)
    #text = kanji2int(text)
    text = ''.join(filter(str.isalnum, text))
    text = re.sub(r"[a-zA-Z]", " ", text)　#  英単語を削除
    node = tagger2.parseToNode(text)
    terms = []
    while node:
        # 単語
        term = node.surface
        # 品詞
        pos = node.feature.split(',')[0]
        # もし品詞が条件と一致してたら
        if pos in select_conditions :
          if node.feature.split(",")[0] in normalize_list:
            term = node.feature.split(",")[6]
          if not term in stopword_list:
            terms.append(term)  
        node = node.next    
    # 連結
    text = ' '.join(terms)
    text = text.split(' ')    
    # 未収録の単語をディクショナリに格納
    for word in text:
      if word not in word_to_id: # 未収録の単語のとき
            # 次の単語のidを取得
        new_id = len(word_to_id)          
            # 単語をキーとして単語IDを格納
        word_to_id[word] = new_id          
            # 単語IDをキーとして単語を格納
        id_to_word[new_id] = word  
          # 単語IDリストを作成
      corpus.append(word_to_id[word])
  return corpus, word_to_id, id_to_word # (受け取るのに3つの変数が必要！)

  # 文書を分かち書き
def wakati_docs(docs):
  tagger2.parse('')
  ans = []
  for idx in range(len(docs)):
    # 前処理
    text = str(docs[idx])
    text = text.lower() # 小文字に変換
    text = unicodedata.normalize("NFKC", text)
    #text = kanji2int(text)
    text = ''.join(filter(str.isalnum, text))
    text = re.sub(r"[a-zA-Z]", " ", text)  #  英単語を削除
    node = tagger2.parseToNode(text)
    terms = []
    while node:
        # 単語
        term = node.surface
        # 品詞
        pos = node.feature.split(',')[0]
        # もし品詞が条件と一致してたら
        if pos in select_conditions:
          if node.feature.split(",")[0] in normalize_list:
            term = node.feature.split(",")[6]
          if not term in stopword_list:
            terms.append(term)
        node = node.next
    # 連結
    text_result = ' '.join(terms)
    text = text_result.split()
    if(len(text) != 0):
      ans.append(text)
  return ans

def inclusive_index(lst, purpose): # 部分一致のindexを返す
    lst = wakati_docs(lst)
    ans = []
    for i, sentence in enumerate(lst):
      for term in sentence:
        if term == purpose:
          ans.append(i)
    return ans
  
# 分かち書きした文書(docs)を単語ごとにidに変換
def input_docs(docs, word2id):
  X = []
  for doc in docs:
    tmp = []
    for term in doc:
      try:
        tmp.append(word2id[term])
      except KeyError as e:
        print(term)
        word2id[term] = len(word2id)
        tmp.append(word2id[term])
    X.append(tmp)
  V = [len(word2id)]
  return [X], list(V)

# input_tagsのJTM用（補助情報にもIDを付与）
def input_docs_tags(docs, tags, word2id, types):
  X = [[],[]]
  for  doc in docs:
    tmp = []
    for term in doc:
      tmp.append(word2id[term])
    X[0].append(tmp)
  for tag in tags:
    tmp = []
    for n in range(len(tag)):
      if tag[n] != 0:
        tmp.append(n)
    X[1].append(tmp)
  V = [len(word2id), len(types)]
  return X, list(V)

# 文書集合から単語文書行列を作成
def bag_of_words(docs, word2id):
  BoW = np.zeros((len(docs), len(word2id)))
  for idx in range(len(docs)):
    for term in docs[idx]:
      BoW[idx][word2id[term]] += 1
  return BoW

# 出現頻度で単語を絞り込む
def transform_word2id(word2id, BoW, min_freq, max_freq): #未完成
  term_sum = BoW.sum(axis=0)
  for i, n in enumerate(term_sum):
    if n < min_freq or n > max_freq:
      BoW[:, i] = np.zeros((1, len(BoW)))
  return word2id, BoW

# 授業科目ごとに，各授業形態の出現頻度をカウント
def count_df(df, types, t_list):
  lst = []
  for j in range(len(types)):
    lst = inclusive_index(df, types[j])
    for i in lst:
      try:
        t_list[i][j]+=1
      except IndexError as e:
        print(types[j], lst)

# 実行時間の計算用
def time2MinSec(time):
  minute = int(time//60)
  second = int(time%60)
  return minute, second
