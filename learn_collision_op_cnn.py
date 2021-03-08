import numpy as np
import os
import Utilities

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D
import tensorflow.keras.backend as be

import random
import numpy as np
from datetime import datetime

class SaveEpochInfo(keras.callbacks.Callback):
    def __init__(self):
        outfile.write(f"training samples: {sol_data_train.shape[0]}, validation samples: {sol_data_test.shape[0]}, solution size: {solsize}, batch size: {batch_size}\n\n")
        outfile.write(f"epoch\tloss\t\t\taccuracy\tval loss\t\tval accuracy\n")
        outfile.flush()

    def on_epoch_end(self, epoch, logs):
        infoStr = f"{epoch:03d}\t{logs['loss']:.6f}\t\t{logs['accuracy']:.4f}\t\t{logs['val_loss']:.6f}\t\t{logs['val_accuracy']:.4f}\n"
        outfile.write(infoStr)
        outfile.flush()

# main entry point start from here
MM = 41
Mtrim = 5
dms=MM-2*Mtrim

settings = Utilities.LoadSettings()
print(settings)
learn_type = settings['learn_type']
epochs_num = int(settings['epochs'])
hidden_layer_num = int(settings['hidden_layers'])
code_len = int(settings['code_len'])
batch_size = int(settings['batch_size'])
noise = int(settings["noise"])
retrain = int(settings["retrain"])
loss_fn = settings["loss_fn"]

print("Loading solution and collision datasets")
#sol_data = Utilities.LoadPickleSolData("Data/full_sol_data.pk")
#coll_data = Utilities.LoadPickleSolData("Data/full_coll_data.pk")
sol_data = Utilities.LoadPickleSolData("Data/small_sol_data_MM_41_MT_5_CT_0.3.pk")
coll_data = Utilities.LoadPickleSolData("Data/small_coll_data_MM_41_MT_5_CT_0.3.pk")

#sol_data = Utilities.LoadPickleSolData("Data/cleaned_sol_data.pk")
#coll_data = Utilities.LoadPickleSolData("Data/cleaned_coll_data.pk")

solsize = sol_data.shape[1]
print(f"solution size: {solsize}")
dms = int(np.cbrt(solsize))

# Scale data in the range of 0 and 1
#print("Scale data in the range of 0 and 1")
#sol_data = (sol_data - np.min(sol_data))/np.max(sol_data)
#coll_data = (coll_data - np.min(coll_data))/np.max(coll_data)

# Split data into the training and test datasets
print("Split data into the training and test datasets")
data_test_size = int(sol_data.shape[0]*0.1)
sol_data_test = sol_data[:data_test_size, :]
sol_data_train = sol_data[data_test_size:, :]
coll_data_test = coll_data[:data_test_size, :]
coll_data_train = coll_data[data_test_size:, :]

print(f"sol_data_train size: {sol_data_train.shape}, sol_data_test size: {sol_data_test.shape}")
print(f"coll_data_train size: {coll_data_train.shape}, coll_data_test size: {coll_data_test.shape}")

if (retrain == 0):
    # Construct CNN model
    print("Construct CNN model")
    learnColOp = Sequential()
    learnColOp.add(Conv3D(filters=4, kernel_size=(5,5,5), padding="same", data_format="channels_last",
                    activation="relu", input_shape=(dms,dms,dms,1)))
    learnColOp.add(keras.layers.BatchNormalization())

    learnColOp.add(Conv3D(filters=8,kernel_size=(3,3,3),padding="same",data_format="channels_last"))
    learnColOp.add(keras.layers.PReLU())

    learnColOp.add(MaxPooling3D(pool_size=(2,2,2)))
    learnColOp.add(Conv3D(filters=16,kernel_size=(3,3,3),padding="same",data_format="channels_last"))
    learnColOp.add(keras.layers.PReLU())

    learnColOp.add(MaxPooling3D(pool_size=(2,2,2)))
    learnColOp.add(Conv3D(filters=32,kernel_size=(3,3,3),padding="same",data_format="channels_last"))
    learnColOp.add(keras.layers.PReLU())

    learnColOp.add(Flatten())
    learnColOp.add(Dense(dms*dms*dms))
    learnColOp.add(keras.layers.PReLU())

    savedModelPath = f"{learn_type}-{loss_fn}-HL{hidden_layer_num}" + "-" + datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    Utilities.RemoveSavedModels(savedModelPath)
    # Create an output file for writing info during the training
    outfile = open(savedModelPath + "/output.txt", "w")
else:
    savedModelPath = f"learn_coll_op_cnn-MAE-03_07_2021-23_03_06"
    outfile = open(savedModelPath + "/retrain_output.txt", "w")

weight_files = savedModelPath + "/BestLearnColOpModel.hdf5"

# Define callbacks
mkcheckpoint=ModelCheckpoint(weight_files, monitor='val_loss',
                             verbose=0, save_best_only=True,
                             mode='auto', period=1)
earlystop = EarlyStopping(monitor='val_loss',min_delta=0,patience=50,verbose=0,mode='auto',
                          baseline=None,restore_best_weights=True)

saveEpochInfoCb = SaveEpochInfo()

# Optimizer type
adamax = keras.optimizers.Adamax(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

if (retrain == 0):
    if loss_fn == "MAE":
        learnColOp.compile(optimizer=adamax,loss='mean_absolute_error',metrics=['accuracy'])
    else:  
        learnColOp.compile(optimizer=adamax,loss='mean_squared_error',metrics=['accuracy']) 
else:   
   learnColOp = keras.models.load_model(savedModelPath + "/LearnColOpModel.hdf5")
   be.set_value(learnColOp.optimizer.lr, 0.001)

# reshape datasets
print("reshape datasets")
sol_data_train = np.reshape(sol_data_train, (len(sol_data_train), dms,dms,dms,1))
sol_data_test = np.reshape(sol_data_test, (len(sol_data_test), dms,dms,dms,1))
coll_data_train = np.reshape(coll_data_train, (len(coll_data_train), dms*dms*dms))
coll_data_test = np.reshape(coll_data_test, (len(coll_data_test), dms*dms*dms))

# compile the model and do the training
print("Compile the model and do the training")
learnColOp.compile(optimizer=adamax,loss='mean_absolute_error',metrics=['accuracy'])
history = learnColOp.fit(sol_data_train, coll_data_train,epochs=epochs_num,batch_size=batch_size,shuffle=True,verbose=1,
                          validation_data=(sol_data_test,coll_data_test),callbacks=[mkcheckpoint, saveEpochInfoCb])

print("Done with the training, saving the model")
# SAVE weights and model
learnColOp.save_weights(savedModelPath + '/LearnColOpCNNWeights.hdf5')
learnColOp.save(savedModelPath + "/LearnColOpCNNModel.hdf5")

outfile.close()
