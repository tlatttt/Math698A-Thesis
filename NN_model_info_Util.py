##########################################################################################################
# Math 689: Thesis
# Student: Thomas V Nguyen
# Advisor: Prof. Alekseenko
#
# This module is to provide functions to display various neural-network information such as:
#   Numer of layers and layer information
#   Plot the history of the training results
#   etc.
#
##########################################################################################################
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

def ModelSummary(model):
    print(model.summary())

def ModelLayers(model):
    print(model.layers)

def LayerWeights(layer):
    weights, biases = layer.get_weights()
    print("weight information:")
    print(weights)
    print("Biases Information: ")
    print(biases)

def PlotTrainingHistory(history):
    pd.DataFrame(history.history).plot()
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

# Testing code
path = input("Enter saved model HDF5 file (q to quit): ")
model = keras.models.load_model(path)
ModelSummary(model)
ModelLayers(model)

