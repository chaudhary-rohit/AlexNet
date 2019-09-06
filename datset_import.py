import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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

#Check random images and labels
for i in range(10):
    a= np.random.randint(10,50+1)
    img = set_xx[a].astype('uint8')
    label = set_yy[a]
    plt.imshow(img)
    plt.show()
    print(label)