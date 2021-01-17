##########################################################################################################
# Math 689: Thesis
# Student: Thomas V Nguyen
# Advisor: Prof. Alekseenko
#
# This is a main module that constructs the auto encoders with different hidden layers
# and train the auto encoders using the solution data. There is a settings.txt file that a user
# can change the settings such that number of hidden layers or code length. This module open this 
# settings.txt file and obtain the setting information to create different auto encoders.
##########################################################################################################

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import pandas as pd
from sklearn.model_selection import train_test_split
import Utilities

import NN_model_info_Util

import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GaussianNoise
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def BuildDeepAutoEncoderModel():
    encoder = keras.models.Sequential()
    decoder = keras.models.Sequential()

    encoder.add(keras.layers.Input(solsize))

    if (hidden_layer_num == 1):
        encoder.add(keras.layers.Dense(code_len, activation='relu'))
        
        decoder.add(keras.layers.Dense(solsize, activation='relu', input_shape=[code_len]))
    elif (hidden_layer_num == 3):
        encoder.add(keras.layers.Dense(2*code_len, activation='relu'))
        encoder.add(keras.layers.Dense(code_len, activation='relu'))
        
        decoder.add(keras.layers.Dense(2*code_len, activation='relu', input_shape=[code_len]))
        decoder.add(keras.layers.Dense(solsize, activation='relu'))
    else:
        encoder.add(keras.layers.Dense(4*code_len, activation='relu'))
        encoder.add(keras.layers.Dense(2*code_len, activation='relu'))
        encoder.add(keras.layers.Dense(code_len, activation='relu'))
        
        decoder.add(keras.layers.Dense(2*code_len, activation='relu', input_shape=[code_len]))
        decoder.add(keras.layers.Dense(4*code_len, activation='relu'))
        decoder.add(keras.layers.Dense(solsize, activation='relu'))

    autoencoder = keras.models.Sequential([encoder, decoder])
    return encoder, decoder, autoencoder
    

settings = Utilities.LoadSettings()
if (int(settings['hidden_layers']) not in [1, 3, 5]):
    print("invalid hidden layer number, the number must be 1, 3, or 5")
    exit(1)

print(settings)
learn_type = settings['learn_type']
epochs_num = int(settings['epochs'])
hidden_layer_num = int(settings['hidden_layers'])
code_len = int(settings['code_len'])
batch_size = int(settings['batch_size'])
noise = int(settings["noise"]) 

sol_data = Utilities.LoadPickleSolData("Data/cleaned_sol_data.pk")
sol_data_train, sol_data_val = train_test_split(sol_data, test_size = 0.1)

solsize = sol_data_train.shape[1]
print(f"solution train data: {sol_data_train.shape}")
print(f"solution validation data: {sol_data_val.shape}")

encoder, decoder, autoencoder = BuildDeepAutoEncoderModel()

if (noise == 0):
    savedModelPath = f"{learn_type}-HL{hidden_layer_num}-CL{code_len}"
else:
    savedModelPath = f"{learn_type}-{noise}%-noise-HL{hidden_layer_num}-CL{code_len}"

# clean up folder before training
Utilities.RemoveSavedModels(savedModelPath)

#### DEFINING CALLBACKS:
weight_files = savedModelPath + "/TLWeights.{epoch:03d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5"
mkcheckpoint=ModelCheckpoint(weight_files, monitor='val_loss',
                             verbose=0, save_best_only=True, save_weights_only=True,
                             mode='auto', period=1)

mkearlystopping = EarlyStopping(patience=50, restore_best_weights=True)

# Choose optimizer
nadam = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-6, schedule_decay=0.001)

# Compile autoencoder
autoencoder.compile(optimizer=nadam,loss='mean_absolute_error',metrics=['accuracy'])

# Train the model
if (noise == 0):
    history = autoencoder.fit(sol_data_train, sol_data_train,
                            epochs=int(settings['epochs']),
                            batch_size=int(settings['batch_size']),
                            shuffle=True,
                            verbose=1,
                            validation_data=(sol_data_val,sol_data_val),callbacks=[mkcheckpoint])
else:
    sol_data_train_with_noise = sol_data_train*(1 + (noise/100.0)*np.random.rand(sol_data_train.shape[0], sol_data_train.shape[1]))
    sol_data_val_with_noise = sol_data_val*(1 + (noise/100.0)*np.random.rand(sol_data_val.shape[0], sol_data_val.shape[1]))
    history = autoencoder.fit(sol_data_train_with_noise, sol_data_train,
                            epochs=int(settings['epochs']),
                            batch_size=int(settings['batch_size']),
                            shuffle=True,
                            verbose=1,
                            validation_data=(sol_data_val_with_noise,sol_data_val),callbacks=[mkcheckpoint])

hist_pd = pd.DataFrame(history.history)
hist_pd.to_csv(f"{savedModelPath}/history.csv")
ax = hist_pd.plot()
title = f'Deep AE Learning Curve: hidden layers:{hidden_layer_num}, code length:{code_len}'
ax.set_title(title)
learning_curve_file = f"{savedModelPath}/LearingCurve-HL{hidden_layer_num}-CL{code_len}.png"
ax.figure.savefig(learning_curve_file)
#plt.show()

# SAVE weights and model
autoencoder.save_weights(savedModelPath + '/Weights.hdf5')
autoencoder.save(savedModelPath + "/Model.hdf5")
