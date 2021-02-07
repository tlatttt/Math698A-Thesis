##########################################################################################################
# Math 689: Thesis
# Student: Thomas V Nguyen
# Advisor: Prof. Alekseenko
#
# This module is to read the solution files from a folder that contains all the solutions and save
# the solution data and collision data files.
# These pickle files will be loaded by another modulew to do the model training and model evaluation.
# 
##########################################################################################################
 
import numpy as np
import my_readwrite
import os
import pickle
import Utilities

def LoadSolCollData(path):
	names = my_readwrite.my_get_soltn_file_names_time(path, cutoff_time)
	sol_data_train, coll_data_train, solsize = my_readwrite.my_read_sol_coll_trim(names[0],MM, Mtrim)  # this is the first file
	zz = len(names)
	num_samples = int(len(names))
	#num_samples = 100
	for i in range(1,num_samples,1):	
		if (i % 10) == 0:
    			print("Proscessing " + str(i) + " of " + str(num_samples))
		solarry, collarry, solsize1 = my_readwrite.my_read_sol_coll_trim(names[i],MM, Mtrim)
		assert solsize1==solsize  # a check that all arrays have the same size -- should be true, but still..
		#sol_max_val = np.amax(solarry)
		#solarry = solarray/sol_max_val
		sol_data_train = np.concatenate((sol_data_train, solarry), axis = 0)
		coll_data_train = np.concatenate((coll_data_train, collarry), axis=0)

	return sol_data_train, coll_data_train

def CreateSolColData(solFilePath, collFilePath):	
	sol_data_train, coll_data_train = LoadSolCollData(path='../alexdata/sphomruns/')
	sol_train_file = open(solFilePath, "wb")
	pickle.dump(sol_data_train, sol_train_file, protocol=4)
	sol_train_file.close()
	coll_train_file = open(collFilePath, "wb")
	pickle.dump(coll_data_train, coll_train_file, protocol=4)
	coll_train_file.close()

def LoadSolData(path):
    names = my_readwrite.my_get_soltn_file_names_time(path, cutoff_time)
    sol_data, solsize = my_readwrite.my_read_solution_trim(names[0],MM, Mtrim)
    num_samples=int(len(names))
	#num_samples = 100
    for i in range(1,num_samples,1):
        if (i % 10) == 0:
            print("Proscessing " + str(i) + " of " + str(num_samples) + " files for dataset")
        solarry, solsize1 = my_readwrite.my_read_solution_trim(names[i],MM, Mtrim)
        assert solsize1==solsize  # a check that all arrays have the same size -- should be true, but still..
	    #sol_max_val = np.amax(solarry)
	    #solarry = solarray/sol_max_val
        sol_data = np.concatenate((sol_data, solarry), axis = 0)

    print(sol_data.shape)
    return sol_data

def CreateSolData(solFilePath):
    solData = LoadSolData(path='../alexdata/sphomruns/')
    sol_file = open(solFilePath, "wb")
    pickle.dump(solData, sol_file, protocol=4)
    sol_file.close()

MM = 41
Mtrim = 5
cutoff_time = 0.30

sol_data_file = f"Data/sol_data_MM_{MM}_MT_{Mtrim}_CT_{cutoff_time}.pk"
coll_data_file = f"Data/coll_data_MM_{MM}_MT_{Mtrim}_CT_{cutoff_time}.pk"
CreateSolColData(sol_data_file, coll_data_file)
#CreateSolData(sol_data_file)