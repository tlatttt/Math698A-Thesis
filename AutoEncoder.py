##########################################################################################################
# Math 689: Thesis
# Student: Thomas V Nguyen
# Advisor: Prof. Alekseenko
#
# This is a main module that constructs the deep and convolutional auto encoders with different hidden
# layers and train the auto encoders using the solution data. There is a settings.txt file that a user
# can change the settings such that number of hidden layers or code length. This module open this 
# settings.txt file and obtain the setting information to create different auto encoders.
##########################################################################################################

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import pandas as pd
import Utilities

import NN_model_info_Util

import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

def LoadSettings():
    settings = {}
    with open("settings.txt") as f:
        for line in f:
            (key, val) = line.split()
            settings[key] = val
    return settings

def BuildDeepAutoEncoderModel():
    if (hidden_layer_num == 1):
        encoder = keras.models.Sequential([
            keras.layers.Input(solsize, 1),
            keras.layers.Dense(code_len, activation='relu')
        ])
        
        decoder = keras.models.Sequential([
            keras.layers.Dense(solsize, activation='relu', input_shape=[code_len]),
            keras.layers.Reshape([solsize,])
        ])
    elif (hidden_layer_num == 3):
        encoder = keras.models.Sequential([
            keras.layers.Input(solsize, 1),
            keras.layers.Dense(2*code_len, activation='relu'),
            keras.layers.Dense(code_len, activation='relu')
        ])
        
        decoder = keras.models.Sequential([
            keras.layers.Dense(2*code_len, activation='relu', input_shape=[code_len]),
            keras.layers.Dense(solsize, activation='relu'),
            keras.layers.Reshape([solsize,])
        ])
    else:
        encoder = keras.models.Sequential([
            keras.layers.Input(solsize, 1),
            keras.layers.Dense(4*code_len, activation='relu'),
            keras.layers.Dense(2*code_len, activation='relu'),
            keras.layers.Dense(code_len, activation='relu')
        ])
        
        decoder = keras.models.Sequential([
            keras.layers.Dense(2*code_len, activation='relu', input_shape=[code_len]),
            keras.layers.Dense(4*code_len, activation='relu'),
            keras.layers.Dense(solsize, activation='relu'),
            keras.layers.Reshape([solsize,])
        ])

    autoencoder = keras.models.Sequential([encoder, decoder])
    return encoder, decoder, autoencoder
    

settings = LoadSettings()
if (int(settings['hidden_layers']) not in [1, 3, 5]):
    print("invalid hidden layer number, the number must be 1, 3, or 5")
    exit(1)

print(settings)
AEtype = settings['autoencoder_type']
epochs_num = int(settings['epochs'])
hidden_layer_num = int(settings['hidden_layers'])
code_len = int(settings['code_len'])
batch_size = int(settings['batch_size'])

sol_data_train = Utilities.LoadPickleSolData("Data/sol_data_train.pk")
sol_data_val = Utilities.LoadPickleSolData("Data/sol_data_val.pk")
solsize = sol_data_train.shape[1]

encoder, decoder, autoencoder = BuildDeepAutoEncoderModel()

#### DEFINING CALLBACKS:
savedModelPath = f"{Utilities.saved_model_dir}-{AEtype}-HL{hidden_layer_num}-CL{code_len}"
Utilities.RemoveSavedModels(savedModelPath)
weight_files = savedModelPath + "/TLWeights.{epoch:03d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5"
mkcheckpoint=ModelCheckpoint(weight_files, monitor='val_loss',
                             verbose=0, save_best_only=True, save_weights_only=True,
                             mode='auto', period=1)

mkearlystopping = EarlyStopping(patience=10, restore_best_weights=True)

# Choose optimizer
nadam = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-3, schedule_decay=0.1)

# Compile autoencoder
autoencoder.compile(optimizer=nadam,loss='mean_absolute_error',metrics=['accuracy'])

# Train the model
history = autoencoder.fit(sol_data_train, sol_data_train,
                        epochs=int(settings['epochs']),
                        batch_size=int(settings['batch_size']),
                        shuffle=True,
                        verbose=1,
                        validation_data=(sol_data_val,sol_data_val),callbacks=[mkcheckpoint, mkearlystopping])

ax = pd.DataFrame(history.history).plot()
if (AEtype == 'DeepAE'):
    title = f'Deep AE Learning Curve: hidden layers:{hidden_layer_num}, code length:{code_len}'
else:
    title = f'CNN auto encoder learning curve'
ax.set_title(title)
learning_curve_file = f"{savedModelPath}/LearingCurve-HL{hidden_layer_num}-CL{code_len}.png"
ax.figure.savefig(learning_curve_file)
#plt.show()

# SAVE weights and model
autoencoder.save_weights(savedModelPath + '/Weights.hdf5')
autoencoder.save(savedModelPath + "/Model.hdf5")
