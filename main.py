import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os, sys
import soundfile as sf
import time
from keras.callbacks import ModelCheckpoint
from livelossplot    import PlotLossesKeras
from keras           import optimizers
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from keras import regularizers

import os, sys
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten
from keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from keras import optimizers
from tensorflow.keras.layers import BatchNormalization
from keras.layers.core import Dropout
import keras.initializers as init

os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess   = tf.compat.v1.Session(config=config)
from keras import backend as K
K.set_session(sess)

#controling_Hyper parameters
batch_size = 50 
nb_classes = 10
nb_epoch   = 100

#Indicate folds
train_fold = [1, 2]
test_fold = 3

str_train_fold = "fold"+str(train_fold[0])+"-"+str(train_fold[1])

#Training data
X_train = np.load( "/folds_mf/2_GTzan_Xs_train_"+str_train_fold+"_110250_75_frozen.npy" )
Y_train = np.load( "/folds_mf/2_GTzan_Ys_train_"+str_train_fold+"_110250_75_frozen.npy" )

# Adapt 1D data to 2D CNN
X_train = np.squeeze(X_train)
X_train = np.expand_dims(X_train, axis = 3)

f = X_train.shape[1]
g = X_train.shape[2]

def model_generator_GTzannet2D_1a():
    
    from keras.layers      import Input, Dense, Conv2D, AveragePooling1D, LeakyReLU, MaxPool2D, Flatten
    from keras.layers.core import Dropout
    from keras.models      import Model
    from tensorflow.compat.v1.keras             import initializers, optimizers, regularizers
    from keras.callbacks   import ModelCheckpoint
    from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
    
    from tensorflow.keras.layers import BatchNormalization
        
    import keras.initializers as init
    
    #from kapre.utils          import Normalization2D
    #from kapre.augmentation   import AdditiveNoise    
    
    sr = 22050
    
    inp   = Input(shape = (f, g, 1)) 
    #----------------------
    conv1  = Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu')(inp)
    norm1  = BatchNormalization()(conv1)
    #----------------------
    conv2  = Conv2D(filters = 32, kernel_size = (3, 3) )(norm1)
    act2   = LeakyReLU(alpha = 0.2)(conv2)
    pool2  = MaxPool2D(pool_size = 2, strides = 2)(act2)
    drop2  = Dropout(0.05)(pool2)
    #----------------------
    conv3  = Conv2D(filters = 64, kernel_size = (3, 3) )(drop2)
    act3   = LeakyReLU(alpha = 0.2)(conv3)
    #----------------------
    conv4  = Conv2D(filters = 64, kernel_size = (3, 3) )(act3)
    act4   = LeakyReLU(alpha = 0.2)(conv4)
    pool4  = MaxPool2D(pool_size = 4, strides = 2)(act4)
    #----------------------
    flat   = Flatten()(pool4)
    #----------------------    
    #dense1 = Dense(1024, activation='relu', kernel_initializer = initializers.glorot_uniform( seed = 0))(flat)
    #drop1  = Dropout(0.80)(dense1)    
    #----------------------    
    #dense2 = Dense(128, activation='relu', kernel_initializer = initializers.glorot_uniform( seed = 0))(flat)
    #drop2  = Dropout(0.80)(dense2)    
     #----------------------    
    dense3 = Dense(1024, activation='relu', kernel_initializer = initializers.glorot_uniform(seed = 0))(flat)
    drop3  = Dropout(0.80)(dense3)    
    #----------------------
    dense4 = Dense(nb_classes, activation='softmax')(drop3)
    #----------------------

    model  = Model(inp, dense4)


    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optimizers.Adadelta(learning_rate = 1.0, rho = 0.95, epsilon = 1e-08, decay = 0.0),
                  metrics = ['accuracy'] )
    
    model.summary()
    
    return model
	
	
hist = []

model = model_generator_GTzannet2D_1a()

#checkpoints
str0 = "weights/"
str1 = "weights_3_GTzan_3f_"+str_train_fold+"_20p_110250_75_frozen_stft" 
str2 = ".best.hdf5" 
filepath = str0+str1+str2
print(filepath)

checkpoint = ModelCheckpoint( filepath, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max' ) 

callbacks_list = [checkpoint, PlotLossesKeras()]

#fitting the model 

batch_size = 50
hist.append(model.fit(X_train, Y_train,
                      batch_size = batch_size, 
                      epochs     = 100,
                      verbose    = 1,
                      shuffle    = True,
                      callbacks  = callbacks_list,
                      validation_split = 0.2                   
                     ))