
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import random
from scipy.io import loadmat
get_ipython().run_line_magic('matplotlib', 'inline')

import keras
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.layers import Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Input

from keras import utils as np_utils
from keras.optimizers import Adam
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau

from keras import regularizers
from keras import metrics
from keras import optimizers
from scipy import misc

from keras.callbacks import TensorBoard


# In[ ]:


# load data
data_path = "./data.mat"
data_raw = loadmat(data_path)

train_img = data_raw["train_img"]
test_img = data_raw["test_img"]
train_lbl = data_raw["train_lbl"]


# ## add noise

# In[ ]:


def saltpepper(trn_data): 
    # input is a n*28*28*1 numpy array, reshape each row to a 28*28 matrix
    row = trn_data.shape[0]
    for i in range(row):
        t = trn_data[i]
        t = t.reshape((28, 28))
        val = [1, 0]
        ver_idx = random.randint(4, 15)
        hor_idx = random.randint(4, 15)

        t[ver_idx:(ver_idx+10), hor_idx:(hor_idx+10)] = np.random.choice(val, size=(10, 10))
        
        t = t.reshape((28,28,1))
        trn_data[i] = t 
    return trn_data

noise_factor = 100/255
def gauss(trn_data):
    # input is a n*28*28*1 numpy array, reshape each row to a 28*28 matrix
    row = trn_data.shape[0]
    for i in range(row):
        t = trn_data[i]
        t = t.reshape((28, 28))
        t[4:24, 4:24] = t[4:24, 4:24] + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=(20,20)) 

        t = t.reshape((28,28,1))
        trn_data[i] = t
    return trn_data


# In[ ]:



x_train_nor = np.reshape(train_img[0:30000], (len(train_img[0:30000]), 28, 28, 1))/255
x_train_sp = np.reshape(train_img[20000:50000], (len(train_img[20000:50000]), 28, 28, 1))/255
x_train_gs = np.reshape(train_img[20000:50000], (len(train_img[20000:50000]), 28, 28, 1))/255

ae_x_train_sp1 = saltpepper(x_train_sp)
ae_x_train_sp2 = saltpepper(x_train_sp)
ae_x_train_sp3 = saltpepper(x_train_sp)
ae_x_train_sp = np.concatenate((ae_x_train_sp1, ae_x_train_sp2, ae_x_train_sp3))

ae_x_train_gs1 = np.clip(gauss(x_train_gs), 0., 1.)
ae_x_train_gs2 = np.clip(gauss(x_train_gs), 0., 1.)
ae_x_train_gs = np.concatenate((ae_x_train_gs1, ae_x_train_gs2))

ae_x_lbl = np.concatenate((x_train_nor, x_train_sp, x_train_sp,x_train_sp,x_train_gs,x_train_gs))

ae_x_trn = np.concatenate((x_train_nor, ae_x_train_sp, ae_x_train_gs)) # "features" for auto encoder

print(ae_x_lbl.shape)
print(ae_x_trn.shape)


# ### Train Auto Encoder

# In[ ]:


ae_trn_X, ae_val_X, ae_trn_lb, ae_val_lb = train_test_split(ae_x_trn,
                                                             ae_x_lbl, 
                                                             test_size=0.1, 
                                                             random_state=13)

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 32)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(ae_trn_X, ae_trn_lb,
                epochs=1,
                batch_size=128,
                shuffle=True,
                validation_data=(ae_val_X, ae_val_lb),
                callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])



# ### Test Auto Encoder on the Test Set

# In[ ]:


X_tst = np.reshape(test_img, (len(test_img), 28, 28, 1))/255 

clean_tst = autoencoder.predict(X_tst, batch_size=128)


# ### CNN-Training Labels

# In[ ]:


# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_trn_nor = train_lbl
Y_trn_sp = np.concatenate((train_lbl, train_lbl, train_lbl))
Y_trn_gs = np.concatenate((train_lbl, train_lbl))

Y_trn_nor = np_utils.to_categorical(Y_trn_nor, num_classes = 10)
Y_trn_sp = np_utils.to_categorical(Y_trn_sp, num_classes = 10)
Y_trn_gs = np_utils.to_categorical(Y_trn_gs, num_classes = 10)

Y_trn = np.concatenate((Y_trn_nor, Y_trn_sp, Y_trn_gs))
Y_trn.shape


# ### CNN-Training Features

# In[ ]:


X_trn = autoencoder.predict(ae_x_trn, batch_size=128)
# X_tst is clean_tst
X_trn.shape


# In[ ]:


plt.figure(figsize=(20,10))
for index, image in enumerate(X_trn[random.sample(range(1, 60000), 25)]):
    plt.subplot(5, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)


# ## split data

# In[ ]:


X_trn, X_val, Y_trn, Y_val = train_test_split(X_trn, Y_trn, test_size = 0.1)


# # CNN

# In[ ]:


epochs = 30 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 64
num_classes = 10
import keras
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

input_shape = (28, 28, 1)
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Valid', activation ='relu', input_shape = (28,28,1)))
model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same',  activation ='relu'))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation ='relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation ='relu'))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))


model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-3), metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.000001)


# In[ ]:


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_trn)


# In[ ]:


# Fit the model

history = model.fit_generator(datagen.flow(X_trn,Y_trn, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_trn.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction],)


# In[ ]:


results = model.predict(clean_tst)
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Prediction")
submission = pd.concat([pd.Series(range(1,20001),name = "ID"),results],axis = 1)

submission.to_csv("yujia.csv",index=False)

