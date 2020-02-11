# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:28:48 2020

@author: Tom Nguyen
"""
import os
import pickle

saved_model_path = 'SavedModels'

def RemoveSavedModels():
	files = os.listdir(saved_model_path)
	for file in files:
		os.remove(saved_model_path + '/' + file)
		
def LoadPickleSolData():
	dataFile = os.getcwd() + "/Data/trimsoldata.pk"
	f = open(dataFile, "rb")
	sol_data = pickle.load(f)
	f.close()
	return sol_data

	