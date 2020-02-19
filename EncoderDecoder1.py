# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:27:35 2020

@author: tlatt
"""
import numpy as np
from sklearn.model_selection import train_test_split
import Utilities
import pandas as pd
import matplotlib.pyplot as plt

saved_model_path = Utilities.saved_model_dir
Utilities.RemoveSavedModels()
sol_data = Utilities.LoadPickleSolData()
  
sol_data_train, sol_data_test = train_test_split(sol_data, test_size = 0.1, random_state = 30)
solsize = sol_data.shape[1]

###### BUILD A SIMPLE AUTOENCODER USING KERAS ##############
#import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

# this is the size of our encoded representation
encoding_dim = 32

#input placeholder
input_vdf = Input(shape=(solsize, ))
# "encoded" is the encoded representation of the input
encoded = Dense(2*encoding_dim, activation='relu')(input_vdf)
encoded = Dense(encoding_dim, activation='relu')(encoded)
encoded = Dense(2*encoding_dim, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(solsize, activation='relu')(encoded)

#this model maps an input to its reconstruction
autoencoder = Model(input_vdf, decoded)

#Let's also create a separate encoder model:
# this model maps the input to its encoded representation
#encoder = Model(input_vdf, encoded)

# We also create a decoder model
#create a placeholder for encoded input
#encoded_input = Input(shape=(encoding_dim, ))
#retrieve the last layer of the autoencoder model
#decoder_layer = autoencoder.layers[-1]
#create the decoder model
#decoder =  Model(encoded_input, decoder_layer(encoded_input))

#### DEFINING CALLBACKS:

mkcheckpoint=ModelCheckpoint('SavedModels/TLWeights.{epoch:03d}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5', monitor='val_loss',
                             verbose=0, save_best_only=True, save_weights_only=True,
                             mode='auto', period=1)
mkearlystopping = EarlyStopping(patience=1, restore_best_weights=True)

#IF FItting from a saved model
#fitting_from_saved_HtNone, decay=0.0)
nadam = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-3, schedule_decay=0.1)
autoencoder.compile(optimizer=nadam,loss='mean_absolute_error',metrics=['accuracy'])
### Check the sizes of training and testing arrays.
print(sol_data_train.shape)
print(sol_data_test.shape)
##################################
### UNCOMMENT IF TRAINING THE MODEL

history = autoencoder.fit(sol_data_train, sol_data_train,epochs=30,batch_size=1,shuffle=True, verbose=1,
                          validation_data=(sol_data_test,sol_data_test),callbacks=[mkcheckpoint])
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

### SAVE weights
autoencoder.save_weights(saved_model_path + '/EncoderDecoder1Weights.hdf5')
autoencoder.save(saved_model_path + "/EncoderDecoder1Model.hdf5")
