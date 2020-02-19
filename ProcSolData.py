# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 13:05:38 2020

@author: Thomas Nguyen

"""
import numpy as np
import my_readwrite
import os
import pickle
import Utilities

def LoadTrimSolData(path, MM, Mtrim):
	names = my_readwrite.my_get_soltn_file_names_time(path, 0.15)
	sol_data, solsize = my_readwrite.my_read_solution_trim(names[0],MM, Mtrim)	
	num_samples=int(len(names))
	#num_samples = 500
	for i in range(1,num_samples,1):
	    print("Proscessing " + str(i) + " of " + str(num_samples) + " files for dataset")
	    solarry, solsize1 = my_readwrite.my_read_solution_trim(names[i],MM, Mtrim)
	    assert solsize1==solsize  # a check that all arrays have the same size -- should be true, but still..
	    #sol_max_val = np.amax(solarry)
	    #solarry = solarray/sol_max_val
	    sol_data = np.concatenate((sol_data, solarry), axis = 0)
	    print(sol_data.shape)
	return sol_data

def saveDataAsPicklefile(file, data):
	f = open(file, "wb")
	pickle.dump(data, f)
	f.close()
	
path = input("Enter the path of solution data: ")

MM = 41
Mtrim = 8
Utilities.CheckRequiredDir()
sol_data = LoadTrimSolData(path, MM, Mtrim)
pickleFile = os.getcwd() + "/Data/trimsoldata.pk"
saveDataAsPicklefile(pickleFile, sol_data)
