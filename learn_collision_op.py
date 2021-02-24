##########################################################################################################
# Math 689: Thesis
# Student: Thomas V Nguyen
# Advisor: Prof. Alekseenko
#
# This module is to train an autoencoder to learn collision operator.
# 
##########################################################################################################

import numpy as np
import os
import Utilities

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GaussianNoise

import random
import numpy as np
from datetime import datetime

class SaveEpochInfo(keras.callbacks.Callback):
    def __init__(self):
        outfile.write(f"training samples: {sol_data_train.shape[0]}, validation samples: {sol_data_test.shape[0]}, solution size: {solsize}\n\n")
        outfile.write(f"epoch\tloss\t\taccuracy\tval loss\tval accuracy\n")

    def on_epoch_end(self, epoch, logs):
        infoStr = f"{epoch:03d}\t{logs['loss']:.4f}\t\t{logs['accuracy']:.4f}\t\t{logs['val_loss']:.4f}\t\t{logs['val_accuracy']:.4f}\n"
        outfile.write(infoStr)
        outfile.flush()

def BuildDenseAutoEncoderModel():
    encoder = keras.models.Sequential()
    decoder = keras.models.Sequential()

    encoder.add(keras.layers.Input(solsize))
    encoder.add(keras.layers.Dropout(0.40))
    keras.layers.BatchNormalization()
    if (hidden_layer_num == 1):
        encoder.add(keras.layers.Dense(code_len, kernel_initializer="he_normal"))
        encoder.add(keras.layers.LeakyReLU())
        encoder.add(keras.layers.BatchNormalization())
        decoder.add(keras.layers.Dense(solsize, kernel_initializer="he_normal", input_shape=[code_len]))
        decoder.add(keras.layers.LeakyReLU())
        
    elif (hidden_layer_num == 3):
        encoder.add(keras.layers.Dense(2*code_len, kernel_initializer="he_normal"))
        encoder.add(keras.layers.BatchNormalization())
        encoder.add(keras.layers.Dense(code_len, kernel_initializer="he_normal"))
        encoder.add(keras.layers.LeakyReLU())
        encoder.add(keras.layers.BatchNormalization())

        decoder.add(keras.layers.Dense(2*code_len, kernel_initializer="he_normal", input_shape=[code_len]))
        decoder.add(keras.layers.LeakyReLU())
        decoder.add(keras.layers.BatchNormalization())
        decoder.add(keras.layers.Dense(solsize, kernel_initializer="he_normal"))
        decoder.add(keras.layers.LeakyReLU())
    elif (hidden_layer_num == 5):
        encoder.add(keras.layers.Dense(4*code_len, kernel_initializer="he_normal"))
        encoder.add(keras.layers.LeakyReLU())
        encoder.add(keras.layers.BatchNormalization())
        encoder.add(keras.layers.Dense(2*code_len, kernel_initializer="he_normal"))
        encoder.add(keras.layers.LeakyReLU())
        encoder.add(keras.layers.BatchNormalization())
        encoder.add(keras.layers.Dense(code_len, kernel_initializer="he_normal"))
        encoder.add(keras.layers.LeakyReLU())
        encoder.add(keras.layers.BatchNormalization())

        decoder.add(keras.layers.Dense(2*code_len, kernel_initializer="he_normal", input_shape=[code_len]))
        decoder.add(keras.layers.LeakyReLU())
        decoder.add(keras.layers.BatchNormalization())
        decoder.add(keras.layers.Dense(4*code_len, kernel_initializer="he_normal"))
        decoder.add(keras.layers.LeakyReLU())
        decoder.add(keras.layers.BatchNormalization())
        decoder.add(keras.layers.Dense(solsize, kernel_initializer="he_normal"))
        decoder.add(keras.layers.LeakyReLU())
    else:
        encoder.add(keras.layers.Dense(8*code_len, kernel_initializer="he_normal"))
        encoder.add(keras.layers.LeakyReLU())
        encoder.add(keras.layers.BatchNormalization())
        encoder.add(keras.layers.Dense(4*code_len, kernel_initializer="he_normal"))
        encoder.add(keras.layers.LeakyReLU())
        encoder.add(keras.layers.BatchNormalization())
        encoder.add(keras.layers.Dense(2*code_len, kernel_initializer="he_normal"))
        encoder.add(keras.layers.LeakyReLU())
        encoder.add(keras.layers.BatchNormalization())
        encoder.add(keras.layers.Dense(code_len, kernel_initializer="he_normal"))
        encoder.add(keras.layers.LeakyReLU())
        encoder.add(keras.layers.BatchNormalization())

        decoder.add(keras.layers.Dense(2*code_len, input_shape=[code_len]))
        decoder.add(keras.layers.LeakyReLU())
        decoder.add(keras.layers.BatchNormalization())
        decoder.add(keras.layers.Dense(4*code_len, kernel_initializer="he_normal"))
        decoder.add(keras.layers.LeakyReLU())
        decoder.add(keras.layers.BatchNormalization())
        decoder.add(keras.layers.Dense(8*code_len, kernel_initializer="he_normal"))
        decoder.add(keras.layers.LeakyReLU())
        decoder.add(keras.layers.BatchNormalization())
        decoder.add(keras.layers.Dense(solsize, kernel_initializer="he_normal"))
        decoder.add(keras.layers.LeakyReLU())

    autoencoder = keras.models.Sequential([encoder, decoder])
    return encoder, decoder, autoencoder


def dataSplit(sol_data, coll_data):
    data_test_size = int(sol_data.shape[0]*0.1)
    sol_data_test = sol_data[:data_test_size, :]
    sol_data_train = sol_data[data_test_size:, :]
    coll_data_test = coll_data[:data_test_size, :]
    coll_data_train = coll_data[data_test_size:, :]

    return sol_data_train, sol_data_test, coll_data_train, coll_data_test


# main entry point start from here
settings = Utilities.LoadSettings()
if (int(settings['hidden_layers']) not in [1, 3, 5, 7]):
    print("invalid hidden layer number, the number must be 1, 3, 5, or 7")
    exit(1)

print(settings)
learn_type = settings['learn_type']
epochs_num = int(settings['epochs'])
hidden_layer_num = int(settings['hidden_layers'])
code_len = int(settings['code_len'])
batch_size = int(settings['batch_size'])
noise = int(settings["noise"])

#sol_data = Utilities.LoadPickleSolData("Data/full_sol_data.pk")
#coll_data = Utilities.LoadPickleSolData("Data/full_coll_data.pk")
sol_data = Utilities.LoadPickleSolData("Data/cleaned_sol_data.pk")
coll_data = Utilities.LoadPickleSolData("Data/cleaned_coll_data.pk")

# scale data between 0 and 1
#sol_data = (sol_data - np.min(sol_data))/np.max(sol_data)
#coll_data = (coll_data - np.min(coll_data))/np.max(coll_data)

solsize = sol_data.shape[1]

sol_data_train, sol_data_test, coll_data_train, coll_data_test = dataSplit(sol_data, coll_data)

encoder, decoder, learnColOp = BuildDenseAutoEncoderModel()

#### DEFINING CALLBACKS:
savedModelPath = f"{learn_type}-HL{hidden_layer_num}-CL{code_len}" + "-" + datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
Utilities.RemoveSavedModels(savedModelPath)

# Create an output file for writing info during the training
outfile = open(savedModelPath + "/output.txt", "w")

weight_files = savedModelPath + "/BestLearnColOpModel.hdf5"
mkcheckpoint=ModelCheckpoint(weight_files, monitor='val_loss',
                             verbose=0, save_best_only=True,
                             mode='auto', period=1)

mkearlystopping = EarlyStopping(patience=50, restore_best_weights=True)
saveEpochInfoCb = SaveEpochInfo()

# Choose optimizer
nadam = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-6, schedule_decay=0.001)

# Compile autoencoder
learnColOp.compile(optimizer=nadam,loss='mean_absolute_error',metrics=['accuracy'])

history = learnColOp.fit(sol_data_train, coll_data_train,
                        epochs=epochs_num,
                        batch_size=batch_size,
                        shuffle=True,
                        verbose=1,
                        validation_data=(sol_data_test,coll_data_test),callbacks=[mkcheckpoint, saveEpochInfoCb])

# SAVE weights and model
learnColOp.save_weights(savedModelPath + '/LearnColOpWeights.hdf5')
learnColOp.save(savedModelPath + "/LearnColOpModel.hdf5")

outfile.close()
