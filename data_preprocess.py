#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:34:23 2018

@author: liushenghui
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# comment_classifier.py
#
# Vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Python source code - replace this with a description
# of the code and write the code below this text
#
import pdb
import numpy as np
from collections import Counter
import jieba
import pandas as pd
import tensorflow as tf
import pickle
import random
from matplotlib import pyplot as plt
import keras
from keras import Model
from keras.models import Sequential 
from keras.layers import Dense,Flatten,Dropout, Input ,LSTM
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical 
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB 
from collections import Counter
from sklearn.lda import LDA
"""
'I'm super man'
tokenize:
['I', ''m', 'super', 'man']
"""

"""
词形还原(lemmatizer)，即把一个任何形式的英语单词还原到一般形式，
与词根还原不同(stemmer)，后者是抽取一个单词的词根。
"""
# 创建词汇表
def create_lexicon(train_file):
    def process_file(txtfile):
        with open(txtfile, 'r',encoding='utf8') as f:
            lex = []
            lines = f.readlines()
            #print(lines)
            for i,line in enumerate(lines):
                try:
                    if i%2000 == 0:
                        print(i)
                    content = line.split('\t')[1]
                    words = jieba.lcut(content)
                    lex += words
                except:
                    pass
			print("分词完成")
            return lex

    lex = process_file(train_file)
    #print(len(lex))
#    lemmatizer = WordNetLemmatizer()
#    lex = [lemmatizer.lemmatize(word) for word in lex] # 词形还原(cats -> cat)

    word_count = Counter(lex)
    #print(word_count)
    lex = []
    for word in word_count:
        if word_count[word] < 2000 and word_count[word] > 600:
            lex.append(word)
    return lex

#lex 里保存了文本中出现过的单词

def string_to_vector2(lex, review):
    words = jieba.lcut(review[5:])
    features = np.zeros(len(lex))
    for word in words:
        if word in lex:
            features[lex.index(word)] += 1
    return features

###会爆内存
def normalize_dataset(lex,file):
    dataset = []
    # lex:词汇表；review:评论；clf:评论对应的分类，
    # [0,1]代表负面评论 [1,0]代表正面评论
    def string_to_vector(lex, review):
        lab, content = review.split('\t')
        words = jieba.lcut(content)
        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1
        return features, lab

    with open(file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        lab_str = []
        for i, line in enumerate(lines):
            try:
                if i%2000 == 0:
                    print(i)
                lab, one_sample = string_to_vector(lex, line)
                dataset.append(one_sample)
                lab_str.append(lab)
            except:
                pass
    return dataset,lab_str

def data_2_array(dataset):
    data_x = []
    data_y = []
    for i in range(len(dataset)):
        data_x.append(dataset[i][0])
        data_y.append(dataset[i][1])
    return np.array(data_x),np.array(data_y)
train_file = 'cnews/cnews.train.txt'
test_file = 'cnews/cnews.test.txt'
lex = create_lexicon(train_file)

train_x, train_lab = normalize_dataset(lex, train_file)
labs = Counter(train_x).keys()
lab_dict = zip(labs,[i for i in range(10)])
test_x, test_lab = normalize_dataset(lex, test_file)

x_train = np.array(train_lab)
y_train = np.array([lab_dict[i] for i in train_x])
x_test = np.array(test_lab)
y_test = np.array([lab_dict[i] for i in test_x])

data = {'x_train':x_train,'x_test': x_test,'y_train': y_train,'y_test':y_test}

with open('lex.pkl','wb') as ff:
    pickle.dump(lex,ff)
#把整理好的数据保存到文件，方便使用。到此完成了数据的整理工作
with open('data.pkl', 'wb') as f:
	pickle.dump(data, f)