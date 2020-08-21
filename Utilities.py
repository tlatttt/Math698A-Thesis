##########################################################################################################
# Math 689: Thesis
# Student: Thomas V Nguyen
# Advisor: Prof. Alekseenko
#
# This module provides various utility functions that can be used by other modules.
#
##########################################################################################################

import os
import pickle

saved_model_dir = 'SavedModels'
data_dir = 'Data'

def RemoveSavedModels(dirpath=saved_model_dir):
	if (os.path.exists(dirpath)):
		files = os.listdir(dirpath)
		for file in files:
			os.remove(dirpath + '/' + file)
	else:
    		os.mkdir(dirpath)
    		
    	
    		
def LoadPickleSolData(data_file):
	f = open(data_file, "rb")
	sol_data = pickle.load(f)
	f.close()
	return sol_data

def CheckRequiredDir():
	if (os.path.exists(saved_model_dir) == False):
		os.mkdir(saved_model_dir)
		
	if (os.path.exists(data_dir) == False):
		os.mkdir(data_dir)