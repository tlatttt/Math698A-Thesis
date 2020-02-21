
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import Utilities
import my_readwrite
import my_viz_routines

MM = 41
#Mtrim = 9
Mtrim = 8

saved_model_dir = Utilities.saved_model_dir
sol_data = Utilities.LoadPickleSolData()
num_test_solutions = 3 # number of test_solutions to select

# Random select three solution data
sol_data_train = np.zeros((3, sol_data.shape[1]))
ind = [i for i in range(sol_data.shape[0])]
random.shuffle(ind)
for i in range(num_test_solutions):
	sol_data_train[i] = sol_data[i]


autoencoder = keras.models.load_model(saved_model_dir + '/EncoderDecoder1Model.hdf5')
autoencoder.summary()

decoded_sols = autoencoder.predict(sol_data_train)

for i in range(0,len(sol_data_train)):
	print(np.amax(np.absolute(sol_data_train[i,:]-decoded_sols[i,:])))

uval = np.arange(-3.0,2.99,0.1)
vval = np.arange(-3.0,2.99,0.1)
fig_num = 1

for i in range(0, num_test_solutions - 1):
	solarray, sizenew = my_readwrite.solution_untrim(sol_data_train[i, :], MM - 2 * Mtrim, Mtrim)
	sol_val, umesh, vmesh = my_viz_routines.eval0k2DUV_sol(solarray[0,:], (-3, 3, -3, 3, -3, 3), (MM, MM, MM),
														uval, vval, 0.0)
	
	fig = plt.figure(fig_num)
	ax = plt.axes(projection='3d')
	ax.plot_surface(umesh, vmesh, sol_val, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.set(xlim=(-3.0, 3.0), ylim=(-3.0, 3.0), zlim=(0.0, 3.0))

	solarray, sizenew = my_readwrite.solution_untrim(decoded_sols[i,:], MM - 2 * Mtrim, Mtrim)
	sol_val1, umesh, vmesh = my_viz_routines.eval0k2DUV_sol(solarray[0, :], (-3, 3, -3, 3, -3, 3), (MM, MM, MM),
                                                       uval, vval, 0.0)
	
	fig_num = fig_num + 1
	fig = plt.figure(fig_num)
	ax = plt.axes(projection='3d')
	ax.plot_surface(umesh, vmesh, sol_val1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.set(xlim=(-3.0, 3.0), ylim=(-3.0,3.0), zlim=(0.0,3.0))

	fig_num = fig_num + 1
	fig = plt.figure(fig_num)
	ax = plt.axes(projection='3d')
	ax.plot_surface(umesh, vmesh, sol_val-sol_val1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	
	plt.show()
	
	
	