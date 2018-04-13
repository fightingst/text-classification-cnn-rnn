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
from sklearn.lda import LDA
from collections import Counter
"""
'I'm super man'
tokenize:
['I', ''m', 'super', 'man']
"""

"""
词形还原(lemmatizer)，即把一个任何形式的英语单词还原到一般形式，
与词根还原不同(stemmer)，后者是抽取一个单词的词根。
"""



class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
def my_RNN():
    ipt = Input(shape=(1,2801))
    x = LSTM(128)(ipt)
    x = Dense(10,activation='softmax')(x)
    model = Model(inputs=ipt,outputs=x)
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])  
    model.summary()  
    return model
# 创建词汇表
def data_2_array(dataset):
    data_x = []
    data_y = []
    for i in range(len(dataset)):
        data_x.append(dataset[i][0])
        data_y.append(dataset[i][1])
    return np.array(data_x),np.array(data_y)

if __name__ == '__main__':
    with open('lex.pkl','rb') as f:
        lex = pickle.load(f)
    with open('data.pkl','rb') as f:
        data = pickle.load(f)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    rnn_train_x = np.expand_dims(x_train, axis=1)
    rnn_train_y = to_categorical(y_train)
    rnn_test_x = np.expand_dims(x_test,axis=1)
    rnn_test_y = to_categorical(y_test)
#############这一段是rnn预测#############
    model = my_RNN()
    history = LossHistory()
#    model.fit(rnn_train_x,rnn_train_y,epochs=50,batch_size = 16, validation_data=(rnn_test_x, rnn_test_y),callbacks=[history])
    #model.save_weights('RNN_model.h5')
    model.load_weights('RNN_model.h5')
    acc = model.evaluate(rnn_test_x, rnn_test_y)[1]
    rnn_pre = model.predict(rnn_test_x,batch_size=16)
    rnn_pred = np.argmax(rnn_pre,axis=1)
    rnn_pred_proba = np.max(rnn_pre,axis=1)
    rnn_pred_proba[1000:] = 0
    #print ('RNN AUC: ',str(metrics.roc_auc_score(y_test,rnn_pre)))
    print ('RNN ACC: ',str(acc))
    print ('RNN Recall for each class: ',str(metrics.recall_score(y_test,rnn_pred, pos_label=1, average=None)))
    print ('RNN F1-score for each class: ',str(metrics.f1_score(y_test,rnn_pred, average=None)))
    print ('RNN Precesion for each class: ',str(metrics.precision_score(y_test,rnn_pred, average=None)))
    metrics.confusion_matrix(y_test,rnn_pred)
##########################LDA预测############     
    clf = LDA()
    clf.fit(x_train, y_train)
    lda_pre = clf.predict_proba(x_test)
    lda_pred = np.argmax(lda_pre,axis=1)
    
    print ('lda ACC: ',str(metrics.accuracy_score(y_test,lda_pred)))
    print ('lda Recall for each class: ',str(metrics.recall_score(y_test,lda_pred, average=None)))
    print ('lda F1-score for each class: ',str(metrics.f1_score(y_test,lda_pred, average=None)))
    print ('lda Precesion for each class: ',str(metrics.precision_score(y_test,lda_pred, average=None)))
    metrics.confusion_matrix(y_test,lda_pred)
    
####################朴素贝叶斯预测################    
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    Bayes_pre = gnb.predict_proba(x_test)
    Bayes_pred = np.argmax(Bayes_pre,axis=1)
    print ('Bayes ACC: ',str(metrics.accuracy_score(y_test,Bayes_pred)))
    print ('Bayes Recall for each class: ',str(metrics.recall_score(y_test,Bayes_pred, average=None)))
    print ('Bayes F1-score for each class: ',str(metrics.f1_score(y_test,Bayes_pred, average=None)))
    print ('Bayes Precesion for each class: ',str(metrics.precision_score(y_test,Bayes_pred, average=None)))
    metrics.confusion_matrix(y_test,Bayes_pred)





    








    


