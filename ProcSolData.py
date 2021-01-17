##########################################################################################################
# Math 689: Thesis
# Student: Thomas V Nguyen
# Advisor: Prof. Alekseenko
#
# This module is to read the solution files from a folder that contains all the solutions and save
# the solution data and collision data files.
# These pickle files will be loaded by another modulew to do the model training and model evalution.
# 
##########################################################################################################
 
import numpy as np
import my_readwrite
import os
import pickle
import Utilities

def LoadSolCollData(path, MM, Mtrim, cutoff_time):
	names = my_readwrite.my_get_soltn_file_names_time(path, 0.3)
	sol_data_train, coll_data_train, solsize = my_readwrite.my_read_sol_coll_trim(names[0],MM, Mtrim)  # this is the first file
	zz = len(names)
	num_samples=int(len(names))
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

def SaveSolColData(solFilePath, collFilePath):	
	sol_data_train, coll_data_train = LoadSolCollData(path='../alexdata/sphomruns/', MM=41, Mtrim=5, cutoff_time=0.3)
	sol_train_file = open(solFilePath, "wb")
	pickle.dump(sol_data_train, sol_train_file, protocol=4)
	sol_train_file.close()
	coll_train_file = open(collFilePath, "wb")
	pickle.dump(coll_data_train, coll_train_file, protocol=4)
	coll_train_file.close()

def LoadSolData(path, MM, Mtrim, cutoff_time):
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

def SaveSolData(solFilePath):
    solData = LoadSolData(path='../alexdata/sphomruns/', MM=41, Mtrim=8, cutoff_time=0.15)
    sol_file = open(solFilePath, "wb")
    pickle.dump(solData, sol_file, protocol=4)
    sol_file.close()

SaveSolColData("Data/full_sol_data.pk", "Data/full_coll_data.pk")
#SaveSolData("Data/sol_data.pk")