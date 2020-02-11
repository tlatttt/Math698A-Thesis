# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:28:48 2020

@author: Tom Nguyen
"""
import os
import pickle

saved_model_dir = 'SavedModels'
data_dir = 'Data'

def RemoveSavedModels():
	files = os.listdir(saved_model_dir)
	for file in files:
		os.remove(saved_model_dir + '/' + file)
		
def LoadPickleSolData():
	dataFile = os.getcwd() + "/Data/trimsoldata.pk"
	f = open(dataFile, "rb")
	sol_data = pickle.load(f)
	f.close()
	return sol_data

def CheckRequiredDir():
	if (os.path.exists(saved_model_dir) == False):
		os.mkdir(saved_model_dir)
		
	if (os.path.exists(data_dir) == False):
		os.mkdir(data_dir)