# -*- coding: utf-8 -*-
"""
Created on Fri May 31 19:44:24 2019

@author: Rohit Chaudhary
"""
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

 
W = {
     'W1': tf.get_variable('w1', shape=(11,11,3,96), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
     'W2': tf.get_variable('w2', shape=(5,5,96,256), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
     'W3': tf.get_variable('w3', shape=(3,3,256,384), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
     'W4': tf.get_variable('w4', shape=(3,3,384,384), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
     'W5': tf.get_variable('w5', shape=(3,3,384,256), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
     'W6': tf.get_variable('w6', shape=(256*6*6,4096), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
     'W7': tf.get_variable('w7', shape=(4096,4096), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
     'W8': tf.get_variable('w8', shape=(4096,1000), initializer=tf.contrib.layers.xavier_initializer_conv2d())
     }
B = {
     'B1': tf.get_variable('b1', shape=(96), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
     'B2': tf.get_variable('b2', shape=(256), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
     'B3': tf.get_variable('b3', shape=(384), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
     'B4': tf.get_variable('b4', shape=(384), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
     'B5': tf.get_variable('b5', shape=(256), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
     'B6': tf.get_variable('b6', shape=(4096), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
     'B7': tf.get_variable('b7', shape=(4096), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
     'B8': tf.get_variable('b8', shape=(1000), initializer=tf.contrib.layers.xavier_initializer_conv2d())
    }

def Conv_Net(x, W, B):
    
    #convolution layer 1
    conv1 = tf.nn.conv2d(x, W['W1'], strides = (1,4,4,1), padding = 'VALID')
    conv1 = tf.nn.bias_add(conv1, B['B1'])
    relu1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(relu1, ksize = (1,3,3,1), strides = (1,2,2,1), padding = 'VALID')
         #normalize?
         #centering?
    #convolution layer 2
    conv2 = tf.nn.conv2d(pool1, W['W2'], strides = (1,1,1,1), padding = 'SAME')
    conv2 = tf.nn.bias_add(conv2, B['B2'])
    pool2 = tf.nn.max_pool(conv2, ksize = (1,3,3,1), strides = (1,2,2,1), padding = 'VALID')
    relu2 = tf.nn.relu(pool2)
         #normalize?
    #convolution layer 3
    conv3 = tf.nn.conv2d(relu2, W['W3'], strides = (1,1,1,1), padding = 'SAME')
    conv3 = tf.nn.bias_add(conv3, B['B3'])
    relu3 = tf.nn.relu(conv3)
    
    #convolution layer 4
    conv4 = tf.nn.conv2d(relu3, W['W4'], strides = (1,1,1,1), padding = 'SAME')
    conv4 = tf.nn.bias_add(conv4, B['B4'])
    relu4 = tf.nn.relu(conv4)
    
    #convolution layer 5
    conv5 = tf.nn.conv2d(relu4, W['W5'], strides = (1,1,1,1), padding = 'SAME')
    conv5 = tf.nn.bias_add(conv5, B['B5'])
    relu5 = tf.nn.relu(conv5) 
    pool5 = tf.nn.max_pool(relu5, ksize = (1,3,3,1), strides = (1,2,2,1), padding = 'VALID')
  
    #Reshape input to interact with FC layer weights
    fc_in = tf.reshape(pool5, shape = [-1, 256*6*6])
       
    #FC-1 + ReLU
    fc1 = tf.matmul(fc_in, W['W6'])
    fc1 = tf.add(fc1, B['B6'])
    relu6 = tf.nn.relu(fc1) 
    
    #Drop-out rate = 1 - prob
    drop1 = tf.nn.dropout(relu6, rate = 0.5) 
   
    #FC-2 + ReLU
    fc2 = tf.matmul(drop1, W['W7'])
    fc2 = tf.add(fc2, B['B7'])
    relu7 = tf.nn.relu(fc2) 
    
    #Drop-out rate = 0.5
    drop2 = tf.nn.dropout(relu7, rate = 0.5) 
    
    #FC-3 + ReLU
    fc3 = tf.matmul(drop2, W['W8'])
    fc3 = tf.add(fc3, B['B8'])
    out = tf.nn.relu(fc3) 
    
    return out

a = cv2.imread('dog.jpg',1)
a = cv2.resize(a, (227,227))
a = a.astype('float32')
a = a.reshape(1,227,227,3)
init = tf.global_variables_initializer()
pred = Conv_Net(a,W,B)
with tf.Session() as sess:
    sess.run(init)
    b= sess.run(pred)

