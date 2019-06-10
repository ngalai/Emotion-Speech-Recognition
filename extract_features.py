# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 09:39:51 2019

@author: ngalai
"""
import librosa 
import numpy as np 
# function for extracting feature from data 
def extract_feature(file_name):    
   X, sample_rate = librosa.load(file_name)     
   mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13 ).T,axis=0)      
   return mfccs

# function for coverting feature to array 
def convert_to_array(wav_file):     
    return np.array(extract_feature(wav_file))

