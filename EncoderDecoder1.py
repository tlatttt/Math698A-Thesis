##########################################################################################################
# Math 689: Thesis
# Student: Thomas V Nguyen
# Advisor: Prof. Alekseenko
#
# This module is to construct encoder/decoder to train model for solution files. The solution data are
# saved into three pickle files: 
#   sol_data_train.pk: this is the training data file.
#   sol_data_val.pk: this is the validation data file.
#   sol_data_test.pk: this is the test data file for evaluation.
#   
##########################################################################################################

import numpy as np
from sklearn.model_selection import train_test_split
import Utilities
import pandas as pd
import matplotlib.pyplot as plt

saved_model_path = Utilities.saved_model_dir
Utilities.RemoveSavedModels()
sol_data_train = Utilities.LoadPickleSolData("Data/sol_data_train.pk")
sol_data_val = Utilities.LoadPickleSolData("Data/sol_data_val.pk")
solsize = sol_data_train.shape[1]

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

#### DEFINING CALLBACKS:
mkcheckpoint=ModelCheckpoint('SavedModels/TLWeights.{epoch:03d}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5', monitor='val_loss',
                             verbose=0, save_best_only=True, save_weights_only=True,
                             mode='auto', period=1)
mkearlystopping = EarlyStopping(patience=5, restore_best_weights=True)

# Choose optimizer
nadam = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-3, schedule_decay=0.1)

# Compile autoencoder
autoencoder.compile(optimizer=nadam,loss='mean_absolute_error',metrics=['accuracy'])

# Train the model
history = autoencoder.fit(sol_data_train, sol_data_train,epochs=30,batch_size=1,shuffle=True, verbose=1,
                          validation_data=(sol_data_val,sol_data_val),callbacks=[mkcheckpoint, mkearlystopping])

# SAVE weights and model
autoencoder.save_weights(saved_model_path + '/EncoderDecoder1Weights.hdf5')
autoencoder.save(saved_model_path + "/EncoderDecoder1Model.hdf5")
