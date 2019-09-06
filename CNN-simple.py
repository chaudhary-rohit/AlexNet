#Importing Libraries

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf

#Importing Dataset Function

def Import_DS():
    DIR = 'Small'
    Categories = ['Cat','Dog']
    
    img_array = []
    i_size = 227
    for category in Categories:
        c_num = Categories.index(category)
        DIR_i = os.path.join(DIR, category)
        for img in os.listdir(DIR_i):
            try:
                DIR_ii = os.path.join(DIR_i, img)
                r_img = cv2.imread(DIR_ii, 1)
                r_img = cv2.resize(r_img, (i_size,i_size))
                img_array.append([r_img, c_num])
            except:
                pass
    
    #shuffle T-data
    import random
    random.shuffle(img_array)
    #Got all images and corresponding labels in a list img_array
    return img_array
  
#Defining Architecture of CNN     

#Dictionary of parameters
WnB = {'w1': tf.get_variable('w1', shape=(5,5,3,16 ), initializer = tf.contrib.layers.xavier_initializer()),
       'w2': tf.get_variable('w2', shape=(5,5,16,32), initializer = tf.contrib.layers.xavier_initializer()),
       'w3': tf.get_variable('w3', shape=(57*57*32,32), initializer = tf.contrib.layers.xavier_initializer()),
       'w4': tf.get_variable('w4', shape=(32,2), initializer = tf.contrib.layers.xavier_initializer()),
       'b1': tf.get_variable('b1', shape=(16), initializer=tf.contrib.layers.xavier_initializer()),
       'b2': tf.get_variable('b2', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
       'b3': tf.get_variable('b3', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
       'b4': tf.get_variable('b4', shape=(2), initializer=tf.contrib.layers.xavier_initializer()),
      }
#Convolution_Neural_Nets

def conv_net(x, WnB): 
# First Convolving    
    conv1 = tf.nn.conv2d(x, WnB['w1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, WnB['b1'])
# First Max-Pooling
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')    
# First ReLU    
    ReLU1 = tf.nn.relu(pool1)     
    
# Second Convolving    
    conv2 = tf.nn.conv2d(ReLU1, WnB['w2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, WnB['b2'])
# Second Max-Pooling
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')    
# Second ReLU    
    ReLU2 = tf.nn.relu(pool2)   
    
# Flattening for FC layer   
    flat_for_fc1 = tf.reshape(ReLU2, [-1, WnB['w3'].get_shape().as_list()[0]])
#First FC layer
    fc1 = tf.matmul(flat_for_fc1, WnB['w3'])
    fc1 = tf.add(fc1, WnB['b3'])
# Third ReLU
    fc1 = tf.nn.relu(fc1) 
    
#Second FC layer
    fc2 = tf.matmul(fc1, WnB['w4'])
    fc2 = tf.add(fc2, WnB['b4'])    
    
    return fc2    

#   MAIN PART                                   
img_array = Import_DS()
i_size = 227
l_rate = 0.001
n_classes = 2
   
#both placeholders are of type float
x = tf.placeholder("float", [None,227,227,3])
y = tf.placeholder("float", [None,n_classes])

pred = conv_net(x, WnB)     #predicted scores/prob
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(cost)
    
#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#calculate accuracy across all the given images and average them out. 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
   sess.run(init) 
   for i in range(200):
       t_x,t_y = img_array[i]
       t_x=t_x.reshape(1,227,227,3)
       a=np.zeros((1,2))
       if t_y==0 :
           a=[[1,0]]
       else:
           a=[[0,1]]
       #Run optimization op (backprop).
       opt = sess.run(optimizer, feed_dict={x: t_x, y: a})
       # Calculate batch loss and accuracy                                     
       ##loss, correct_prediction = sess.run([cost, correct_prediction], feed_dict={x: t_x, y: a})
     
    
 
   # Testing 
   test=np.array([])
   for i in range(10):
       test_x, test_y = img_array[i+251]
       test_x=test_x.reshape(1,227,227,3)
       a=np.zeros((1,2))
       if t_y==0 :
           a=[[1,0]]
       else:
           a=[[0,1]]
       correct_p = sess.run(correct_prediction, feed_dict={x: test_x, y: a})
       print(correct_p)
       test=np.append(test, correct_p)
   accuracy = np.mean(test)    
   print('Accuracy = %f',accuracy)    
   sess.close()  
    
    
    
    

    
    
    