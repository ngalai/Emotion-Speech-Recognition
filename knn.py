# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 01:21:33 2019

@author: ngalai
"""

from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from extract_features import extract_feature

def knn_model(): 
    
    fea_gender = np.load('fea_emotion.npy')
    #fea_gender = np.load('fea_test.npy')
    lab_gender = np.load('lab_emotion.npy')
    #lab_gender = np.load('lab_test.npy')
    lab_gender = lab_gender.transpose()
    lab_gender = lab_gender.ravel()

    X_train , X_test, y_train, y_test = train_test_split(fea_gender, lab_gender, test_size = 0.3, random_state = 109)
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, knn.predict(X_train))
    acc_test = accuracy_score(y_test, knn.predict(X_test))
    print('Train acc: %.2f'%acc_train)
    print('Val acc: %.2f'%acc_test)
    return knn

def auto_predict(wav_file):
    result = ['female','male']
    X_pred = extract_feature(wav_file)
    X_pred = X_pred.reshape(-1,13)
    y_pred = knn_model().predict(X_pred)
    return result[int(y_pred)]

knn_model()
#wav_file = 'fe.wav'
#print('Predict Result:', auto_predict(wav_file))