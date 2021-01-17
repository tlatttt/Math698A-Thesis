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

def BuildDenseAutoEncoderModel():
    encoder = keras.models.Sequential()
    decoder = keras.models.Sequential()

    encoder.add(keras.layers.Input(solsize))
    keras.layers.BatchNormalization()
    if (hidden_layer_num == 1):
        encoder.add(keras.layers.Dense(code_len, activation='relu'))
        keras.layers.BatchNormalization()
        decoder.add(keras.layers.Dense(solsize, activation='relu', input_shape=[code_len]))
    elif (hidden_layer_num == 3):
        encoder.add(keras.layers.Dense(2*code_len, activation='relu'))
        keras.layers.BatchNormalization()
        encoder.add(keras.layers.Dense(code_len, activation='relu'))
        keras.layers.BatchNormalization()

        decoder.add(keras.layers.Dense(2*code_len, activation='relu', input_shape=[code_len]))
        keras.layers.BatchNormalization()
        decoder.add(keras.layers.Dense(solsize, activation='relu'))
    else:
        encoder.add(keras.layers.Dense(4*code_len, activation='relu'))
        keras.layers.BatchNormalization()
        encoder.add(keras.layers.Dense(2*code_len, activation='relu'))
        keras.layers.BatchNormalization()
        encoder.add(keras.layers.Dense(code_len, activation='relu'))
        keras.layers.BatchNormalization()

        decoder.add(keras.layers.Dense(2*code_len, activation='relu', input_shape=[code_len]))
        keras.layers.BatchNormalization()
        decoder.add(keras.layers.Dense(4*code_len, activation='relu'))
        keras.layers.BatchNormalization()
        decoder.add(keras.layers.Dense(solsize, activation='relu'))
        
    autoencoder = keras.models.Sequential([encoder, decoder])
    return encoder, decoder, autoencoder


def dataSplit(sol_data, coll_data):
    data_test_size = int(sol_data.shape[0]*0.1)
    sol_data_test = sol_data[:data_test_size, :]
    sol_data_train = sol_data[data_test_size:, :]
    coll_data_test = coll_data[:data_test_size, :]
    coll_data_train = coll_data[data_test_size:, :]

    '''
    shuffled_indices = np.random.permutation(sol_data.shape[0])
    sol_data_test = sol_data[shuffled_indices[:data_test_size], :]
    sol_data_train = sol_data[shuffled_indices[data_test_size:], :]
    coll_data_test = coll_data[shuffled_indices[:data_test_size], :]
    coll_data_train = coll_data[shuffled_indices[data_test_size:], :]
    '''
    return sol_data_train, sol_data_test, coll_data_train, coll_data_test


# main entry point start from here
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
normalize = int(settings["normalize"]) 

#sol_data = Utilities.LoadPickleSolData("Data/full_sol_data.pk")
#coll_data = Utilities.LoadPickleSolData("Data/full_coll_data.pk")
sol_data = Utilities.LoadPickleSolData("Data/cleaned_sol_data.pk")
coll_data = Utilities.LoadPickleSolData("Data/coll_data.pk")

solsize = sol_data.shape[1]

if (normalize == 1):
    sol_data = sol_data - np.min(sol_data)
    coll_data = coll_data - np.min(coll_data) # Make all data to be non-negative
elif (normalize == 2):
    sol_data = (sol_data - np.min(sol_data))/np.max(sol_data)
    coll_data = (coll_data - np.min(coll_data))/np.max(coll_data) # normize using min-max scale

#coll_data_train, coll_data_val = train_test_split(coll_data, test_size = 0.1)
sol_data_train, sol_data_test, coll_data_train, coll_data_test = dataSplit(sol_data, coll_data)

encoder, decoder, learnColOp = BuildDenseAutoEncoderModel()

#### DEFINING CALLBACKS:
savedModelPath = f"{learn_type}-HL{hidden_layer_num}-CL{code_len}"
'''
if (normalize == 1):
    savedModelPath += "-positive"
elif (normalize == 2):
    savedModelPath += "-min-max"
'''
Utilities.RemoveSavedModels(savedModelPath)

weight_files = savedModelPath + "/TLWeights.{epoch:03d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5"
mkcheckpoint=ModelCheckpoint(weight_files, monitor='val_loss',
                             verbose=0, save_best_only=True, save_weights_only=True,
                             mode='auto', period=1)

mkearlystopping = EarlyStopping(patience=50, restore_best_weights=True)

# Choose optimizer
nadam = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-6, schedule_decay=0.001)

# Compile autoencoder
learnColOp.compile(optimizer=nadam,loss='mean_absolute_error',metrics=['accuracy'])

history = learnColOp.fit(sol_data_train, coll_data_train,
                        epochs=epochs_num,
                        batch_size=batch_size,
                        shuffle=True,
                        verbose=1,
                        validation_data=(sol_data_test,coll_data_test),callbacks=[mkcheckpoint])

hist_pd = pd.DataFrame(history.history)
hist_pd.to_csv(f"{savedModelPath}/history.csv")
ax = hist_pd.plot()
title = f'Learning Curve: hidden layers:{hidden_layer_num}, code length:{code_len}'
ax.set_title(title)
learning_curve_file = f"{savedModelPath}/LearingCurve-HL{hidden_layer_num}-CL{code_len}.png"
ax.figure.savefig(learning_curve_file)
#plt.show()

# SAVE weights and model
learnColOp.save_weights(savedModelPath + '/Weights.hdf5')
learnColOp.save(savedModelPath + "/Model.hdf5")
