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

import random
import numpy as np
from datetime import datetime

class SaveEpochInfo(keras.callbacks.Callback):
    def __init__(self):
        outfile.write(f"training samples: {sol_data_train.shape[0]}, validation samples: {sol_data_val.shape[0]}, solution size: {solsize}\n\n")
        outfile.write(f"epoch\tloss\t\taccuracy\tval loss\tval accuracy\n")

    def on_epoch_end(self, epoch, logs):
        infoStr = f"{epoch:03d}\t{logs['loss']:.4f}\t\t{logs['accuracy']:.4f}\t\t{logs['val_loss']:.4f}\t\t{logs['val_accuracy']:.4f}\n"
        outfile.write(infoStr)
        outfile.flush()

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
savedModelPath += "-" + datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
Utilities.RemoveSavedModels(savedModelPath)

# Create an output file for writing info during the training
outfile = open(savedModelPath + "/output.txt", "w")

#### DEFINING CALLBACKS:
weight_files = savedModelPath + "/TLWeights.hdf5"
mkcheckpoint=ModelCheckpoint(weight_files, monitor='val_loss',
                             verbose=0, save_best_only=True, save_weights_only=True,
                             mode='auto', period=1)

mkearlystopping = EarlyStopping(patience=50, restore_best_weights=True)
saveEpochInfoCb = SaveEpochInfo()

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
                            validation_data=(sol_data_val,sol_data_val),callbacks=[mkcheckpoint, saveEpochInfoCb])
else:
    sol_data_train_with_noise = sol_data_train*(1 + (noise/100.0)*np.random.rand(sol_data_train.shape[0], sol_data_train.shape[1]))
    sol_data_val_with_noise = sol_data_val*(1 + (noise/100.0)*np.random.rand(sol_data_val.shape[0], sol_data_val.shape[1]))
    history = autoencoder.fit(sol_data_train_with_noise, sol_data_train,
                            epochs=int(settings['epochs']),
                            batch_size=int(settings['batch_size']),
                            shuffle=True,
                            verbose=1,
                            validation_data=(sol_data_val_with_noise,sol_data_val),callbacks=[mkcheckpoint, saveEpochInfoCb])

hist_pd = pd.DataFrame(history.history)
hist_pd.to_csv(f"{savedModelPath}/history.csv")
ax = hist_pd.plot()
title = f'Deep AE Learning Curve: hidden layers:{hidden_layer_num}, code length:{code_len}'
ax.set_title(title)
learning_curve_file = f"{savedModelPath}/LearingCurve-HL{hidden_layer_num}-CL{code_len}.png"
ax.figure.savefig(learning_curve_file)
#plt.show()

# Do testing for trained model
outfile.write("\nPredict solutions and compute errors\n")
decoded_sols = autoencoder.predict(sol_data_val)
for i in range(0, 50):
    max_sol = np.amax(np.abs(sol_data_val[i,:]))
    max_sol = np.max([max_sol, 1.0e-6])
    error = np.amax(np.absolute(sol_data_val[i,:]-decoded_sols[i,:]))/max_sol
    outfile.write(f"{error:.4f}\n")

# Compute error statistics
outfile.write("\nCompute error statistics\n")
for i in range(0, 50):
    error = np.absolute(sol_data_val[i,:] - decoded_sols[i,:])
    outfile.write(f"mean = {error.mean():.4f}, std = {error.std():.4f}, var = {error.var():.4f}, min = {error.min():.4f}, max = {error.max():.4f}\n")

# SAVE weights and model
autoencoder.save_weights(savedModelPath + '/Weights.hdf5')
autoencoder.save(savedModelPath + "/Model.hdf5")

outfile.close()