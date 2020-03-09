import numpy as np
import os
import matplotlib.pyplot as plt
import Utilities
import pandas as pd

import NN_model_info_Util

import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

# this is the size of our encoded representation
encoding_dim = 32

saved_model_path = Utilities.saved_model_dir
Utilities.RemoveSavedModels()
sol_data_train = Utilities.LoadPickleSolData("Data/sol_data_train.pk")
sol_data_val = Utilities.LoadPickleSolData("Data/sol_data_val.pk")
solsize = sol_data_train.shape[1]

encoder = keras.models.Sequential([
    keras.layers.Input(solsize, 1),
    keras.layers.Dense(2*encoding_dim, activation='relu'),
    keras.layers.Dense(encoding_dim, activation='relu')
])

decoder = keras.models.Sequential([
    keras.layers.Dense(2*encoding_dim, activation='relu', input_shape=[encoding_dim]),
    keras.layers.Dense(solsize, activation='relu'),
    keras.layers.Reshape([solsize,])
])

autoencoder = keras.models.Sequential([encoder, decoder])

#### DEFINING CALLBACKS:
mkcheckpoint=ModelCheckpoint('SavedModels/TLWeights.{epoch:03d}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5', monitor='val_loss',
                             verbose=0, save_best_only=True, save_weights_only=True,
                             mode='auto', period=1)
mkearlystopping = EarlyStopping(patience=5, restore_best_weights=True)

# Choose optimizer
nadam = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-3, schedule_decay=0.1)

# Compile autoencoder
autoencoder.compile(optimizer=nadam,loss='mean_absolute_error',metrics=['accuracy'])

# Train the model
history = autoencoder.fit(sol_data_train, sol_data_train,epochs=30,batch_size=1,shuffle=True, verbose=1,
                          validation_data=(sol_data_val,sol_data_val),callbacks=[mkcheckpoint, mkearlystopping])

# SAVE weights and model
autoencoder.save_weights(saved_model_path + '/EncoderDecoder2Weights.hdf5')
autoencoder.save(saved_model_path + "/EncoderDecoder2Model.hdf5")
