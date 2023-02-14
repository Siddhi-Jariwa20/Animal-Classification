#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt


# In[9]:


all_classes = ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'squirrel', 'spider']
folder_names = ['cane', 'cavallo', 'elefante', 'farfalla','gallina', 'gatto', 'mucca', 'pecora', 'scoiattolo', 'ragno']
X = []
Y = []
for i in range(10):
    path = 'C://Users//ADMIN//Desktop//ANIMAL//raw-img//' + folder_names[i]   
    for file in os.listdir(path):
        try: 
            arr_img = cv2.imread(os.path.join(path, file))
            
#             resizing the images
            arr_img = cv2.resize(arr_img, (100,100))
    
            X.append(arr_img)
            Y.append(i)
        except:
            pass
        
X = np.array(X)
Y = np.array(Y)
# Training and Testing set creation + shuffling them
train_X_set, test_X_set, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 100)


# In[3]:


# normalizing the data to remove the duplicate data
train_X_set = train_X_set / 255.0
test_X_set = test_X_set / 255.0
total_classes = []
for i in range(10):
    total_classes.append(np.count_nonzero(Y==i))

largest_classes = total_classes[0] + total_classes[9]
print('Amount of samples from the two largest classes: ', largest_classes)
print('Ratio to total number of classes: ', largest_classes/sum(total_classes))
total_sum = sum(total_classes)

random_choice = []
for i in range(test_X_set.shape[0]):
    random_choice.append(np.random.choice(np.arange(0, 10), p = [total_classes[0]/total_sum, total_classes[1]/total_sum, total_classes[2]/total_sum, total_classes[3]/total_sum, total_classes[4]/total_sum, total_classes[5]/total_sum, total_classes[6]/total_sum, total_classes[7]/total_sum, total_classes[8]/total_sum, total_classes[9]/total_sum]))
    
print('First 100 selections of random selector: ', random_choice[0:100])

print(classification_report(Y_test, random_choice, target_names = all_classes))


# In[4]:


input_shape = train_X_set[0].shape
no_of_classes = 10
def Training_Function(model):
    history = model.fit(
    x = train_X_set, 
    y = Y_train,
    epochs = 15,
    batch_size = 1000,
    validation_split=0.1,
    verbose=1
    )

    history = pd.DataFrame(history.history)
    display(history)
    
def Testing(model):
    predict_val = model.predict(test_X_set)

    predict_val = np.argmax(predict_val, axis = 1)

    print(classification_report(Y_test, predict_val, target_names = all_classes))
    
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)

k_keras = keras.Sequential()

k_keras.add(keras.layers.Conv2D(32, kernel_size = 3, activation='relu', input_shape = input_shape))
k_keras.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

k_keras.add(keras.layers.Conv2D(64, kernel_size = 3, activation='relu'))
k_keras.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

k_keras.add(keras.layers.Conv2D(128, kernel_size = 3, activation='relu'))
k_keras.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

k_keras.add(keras.layers.Flatten())

k_keras.add(keras.layers.Dense(256, activation = 'relu'))

k_keras.add(keras.layers.Dense(128, activation = 'relu'))

k_keras.add(keras.layers.Dense(64, activation = 'relu'))

k_keras.add(keras.layers.Dense(units=no_of_classes, activation = 'softmax'))

k_keras.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

k_keras.summary()


# In[5]:


#training the model
Training_Function(k_keras)


# In[7]:


#Testing the model
Testing(k_keras)


# In[ ]:




