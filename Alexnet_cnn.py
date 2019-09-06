# ALEXNET ARCHITECTURE CNN MODEL
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Importing Dataset
def Imp_DS():
    
    dir_labels = 'Dataset\Labels'
    dir_jpg = 'Dataset\JPEG'
    label = 0
    set_x = []
    set_y = []
    for label_list in os.listdir(dir_labels):
         label+=1
         dir_l2 = os.path.join(dir_labels, label_list)
         file = open(dir_l2, 'r')
         for line in file:
             thisline = line.split(' ')
             jpg_name = thisline[0]+".jpg"
             label_1 = thisline[1]
             if label_1 == '':
                 label_2 = int(thisline[2])
                 if label_2 == 1:
                     dir_jpg2 = os.path.join(dir_jpg, jpg_name)
                     image_orig = cv2.imread(dir_jpg2, 1)
                     image_resize = cv2.resize(image_orig, (227,227))
                     np_img = image_resize.astype('float32')
                     set_x.append(np_img)
                     set_y.append(label)
         file.close()          
         #Stacking Dataset
    set_xx = np.stack(set_x, axis=0)
    set_yy = np.stack(set_y, axis=0)           

    #Shuffling Data
    rng_state = np.random.get_state()
    np.random.shuffle(set_xx)
    np.random.set_state(rng_state)
    np.random.shuffle(set_yy)

    #One hot encoding
    no_img = len(set_xx)
    encode = np.zeros((no_img,15))
    for i in range(no_img):
        encode[i][set_yy[i]-1] = 1
    
    #Splitting Data into train and test ration 8:2
    x_train = set_xx[0:int(0.8*len(set_xx))]
    y_train = set_yy[0:int(0.8*len(set_yy))]
    
    x_test = set_xx[int(0.8*len(set_xx)):len(set_xx)]
    y_test = set_yy[int(0.8*len(set_yy)):len(set_yy)]
    
    return x_train, y_train, x_test, y_test
    
 
       
# MAIN CONV FEATURE - input image-batch and outputs score
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

# Dictionaries for weights, biases and labels
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
L = {
     '1': 'Aeroplane',
     '2': 'Bicycle',
     '3': 'Bird',
     '4': 'Boat',
     '5': 'Bottle',
     '6': 'Bus',
     '7': 'Car',
     '8': 'Cat',
     '9': 'Chair',
     '10': 'Cow',
     '11': 'Dining Table',
     '12': 'Dog',
     '13': 'Horse',
     '14': 'Motor-Bike',
     '15': 'Person'
     }

# TRAINING PART
def train_test(x_train, y_train, x_test, y_test):
    
    n_classes = 15
    batch_size = 64        #mini-batch gradient descent
    n_train_img = 4500
    n_iter = 100
    learning_rate = 0.001
    # Placeholders for holding batch data
    x = tf.placeholder('float32', shape = [None, 227, 227, 3])
    y = tf.placeholder('float32', shape = [None, n_classes])
        
    # To initialize all global variables W, B
    init = tf.global_variables_initializer()
    
    #operations for training part
    pred = Conv_Net(x,W,B)
    softmax_loss = tf.nn.softmax_cross_entropy_with_logits_v2(y, pred)
    cost = tf.reduce_mean(softmax_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
       
    #to print details like losses and accuracy
    c_predict = tf.equal(tf.arg_max(pred,1), tf.arg_max(y,1))
    accuracy = tf.reduce_mean(tf.cast(c_predict, tf.float32))  
    
    #training starts here
    with tf.Session() as sess
        
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        summary_writer = tf.summary.FileWriter('/alex_savedata', sess.graph)
                
        for i in range(n_iter):
                for batch in range(n_train_img//batch_size)
                    train_x = x_train[batch*batch_size:np.minimum((batch+1)*batch_size,N_train_img)]
                    train_y  = y_train[batch*batch_size:np.minimum((batch+1)*batch_size,N_train_img)]
                    sess.run(optimizer, feed_dict = {x : train_x, y : train_y})
                    #Getting loss and accuracy for current batch
                    loss, acc = sess.run([cost, accuracy], feed_dict = {x : train_x, y : train_y})
                
                print("Epoch :" + str(i) +", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)))
                #Testing on test data for 
                t_loss, t_acc = sess.run([cost, accuracy], feed_dict = {x : x_test, y : y_test})
                
                train_loss.append(loss)
                test_loss.append(valid_loss)
                train_accuracy.append(acc)
                test_accuracy.append(test_acc)
                
                print("Testing Accuracy:","{:.5f}".format(t_acc))

        summary_writer.close()

############# EXECUTION STARTS HERE #############























