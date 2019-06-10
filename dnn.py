# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 20:45:17 2019

@author: ngalai
"""
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.layers import  Flatten, Dropout, Activation
from keras.utils import np_utils
import numpy as np 
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from keras import optimizers


model = Sequential()
model.add(Dense(512,input_dim=13,kernel_initializer='normal'))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(512))
model.add(Activation('sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer= 'adam',metrics=['accuracy'])

'''
model = Sequential()
model.add(Dense(30, input_dim=13, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
'''

# load data for classification 
fea_gender = np.load('fea_emotion.npy')
fea_gender = fea_gender.reshape(-1,13) 

lab_gender = np.load('lab_emotion.npy')
lab_gender = lab_gender.transpose()
lab_gender = lab_gender.ravel()


X_train = fea_gender[:280]
X_val = fea_gender[280:]

y_train = lab_gender[:280]
y_train = np_utils.to_categorical(y_train,2)
y_val = lab_gender[280:]
y_val = np_utils.to_categorical(y_val,2)

model.fit(X_train,y_train,batch_size= 4 ,epochs= 100,verbose=2)
re_loss, re_acc = model.evaluate(X_val,y_val)

print(re_acc)


'''
model_json = model.to_json()
with open("dnn.json", "w") as json_file:
    json_file.write(model_json)
       
# serialize weights to HDF5
model.save_weights("dnn.h5")
print("Saved model to disk")
'''

#####################################
'''
json_file = open('dnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("dnn.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=1)
print("\n")
print('Test accuracy:', score[1] )
'''