# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 23:35:09 2019

@author: ngalai
"""
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import recall_score
from extract_features import convert_to_array

# create a svm classiffier  
def svm_model(): 
    
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    return clf 

# function for predicting any audio file 
def predict_demo(wav_file): 
    
    result = ['anger','neutral']
    X_pred = convert_to_array(wav_file)
    X_pred = X_pred.reshape(-1,13)
    y_pred = svm_model().predict(X_pred)
    return result[int(y_pred)]
    

fea_emotion = np.load('fea_emotion.npy')
lab_emotion = np.load('lab_emotion.npy')
#lab_gender = np.load('lab_test.npy')
lab_emotion = lab_emotion.transpose()
lab_emotion = lab_emotion.ravel()


X_train = fea_emotion[:280]
X_test = fea_emotion[280:]
y_train = lab_emotion[:280]
y_test = lab_emotion[280:]
#X_train , X_test, y_train, y_test = train_test_split(fea_emotion, lab_emotion, test_size = 0.3, random_state = 109)

# test model
'''
y_pred = svm_model().predict(X_test)
# model accuracy
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
recall_average = recall_score(y_test, y_pred, average="binary", pos_label= '0')
print("Recall: ", recall_average)
print("Predict: ",metrics.precision_score(y_test, y_pred, average="binary", pos_label= '0'))
'''

'''
wav_file = 'F79.wav'
x = predict_demo(wav_file)
print(x)
'''