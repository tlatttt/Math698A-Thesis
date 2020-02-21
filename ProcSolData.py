##########################################################################################################
# Math 689: Thesis
# Student: Thomas V Nguyen
# Advisor: Prof. Alekseenko
#
# This module is to read the solution files from a folder that contains all the solutions and save
# the solution data into three files:
#	sol_data_train.pk
#	sol_data_val.pk
#	sol_data_test.pk
# These pickle files will be loaded by another modulew to do the model training and model evalution.
# 
##########################################################################################################
 
import numpy as np
import my_readwrite
import os
import pickle
import Utilities
from sklearn.model_selection import train_test_split

def LoadTrimSolData(path, MM, Mtrim):
	names = my_readwrite.my_get_soltn_file_names_time(path, 0.15)
	sol_data, solsize = my_readwrite.my_read_solution_trim(names[0],MM, Mtrim)	
	num_samples=int(len(names))
	#num_samples = 100
	for i in range(1,num_samples,1):
	    print("Proscessing " + str(i) + " of " + str(num_samples) + " files for dataset")
	    solarry, solsize1 = my_readwrite.my_read_solution_trim(names[i],MM, Mtrim)
	    assert solsize1==solsize  # a check that all arrays have the same size -- should be true, but still..
	    #sol_max_val = np.amax(solarry)
	    #solarry = solarray/sol_max_val
	    sol_data = np.concatenate((sol_data, solarry), axis = 0)
	    print(sol_data.shape)
	return sol_data

def saveDataAsPicklefile(data):
	sol_data_train, sol_data_test = train_test_split(sol_data, test_size = 0.2)
	sol_data_val, sol_data_test = train_test_split(sol_data_test, test_size = 0.5)
    
	train_file = open("Data/sol_data_train.pk", "wb")
	val_file = open("Data/sol_data_val.pk", "wb")
	test_file = open("Data/sol_data_test.pk", "wb")

	pickle.dump(sol_data_train, train_file)
	pickle.dump(sol_data_val, val_file)
	pickle.dump(sol_data_test, test_file)

	train_file.close()
	val_file.close()
	test_file.close()
	
path = input("Enter the path of solution data: ")

MM = 41
Mtrim = 8
Utilities.CheckRequiredDir()
sol_data = LoadTrimSolData(path, MM, Mtrim)
saveDataAsPicklefile(sol_data)
