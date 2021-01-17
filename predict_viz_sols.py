import numpy as np
import random
import my_readwrite
import my_viz_routines
import Utilities

def loadSolData():
    path='C:/Courses/alexdata/sphomruns'
    names = my_readwrite.my_get_soltn_file_names_time (path, 0.15)
    num_test_solutions = int(settings['num_test_solutions']) # number of test_solutions to select
    num_sols = len(names)  # total available solutions
    test_sols_names = []   #prepare an empty array that will keep names of the solutions to study

    for x in range(num_test_solutions):
        ii=random.randint(0,num_sols-1)
        test_sols_names.append(names[ii])

    ### TRIMMING PARAMETERS
    MM = 41
    Mtrim = 0

    # let us load training data, which will be instances of solutions.
    # next we will read some of the files, get solutions out of them and copy solution into
    # an list of arrays. Each array contains one instance of the solution
    if (int(settings["predict_type"]) == 0):
        sol_data_train, solsize = my_readwrite.my_read_solution_trim(test_sols_names[0],MM, Mtrim)  # this is the first file
    else:
        sol_data_train, coll_data_train, solsize = my_readwrite.my_read_sol_coll_trim(test_sols_names[0],MM, Mtrim)

    # let us add more data points/read a few more files
    zz = len(test_sols_names)
    for i in range(zz):
        if (int(settings["predict_type"]) == 0):
            solarry, solsize1 = my_readwrite.my_read_solution_trim(test_sols_names[i], MM, Mtrim)
            assert solsize1==solsize  # a check that all arrays have the same size -- should be true, but still..
            sol_data_train = np.concatenate((sol_data_train, solarry), axis = 0)
        else:
            solarry, collarry, solsize1 = my_readwrite.my_read_sol_coll_trim(test_sols_names[i],MM, Mtrim)
            assert solsize1==solsize  # a check that all arrays have the same size -- should be true, but still..
            sol_data_train = np.concatenate((sol_data_train, solarry), axis = 0)
            coll_data_train = np.concatenate((coll_data_train, collarry), axis=0) 

    if (int(settings["predict_type"]) == 0):
        return sol_data_train, solsize
    else:
        return sol_data_train, coll_data_train, solsize

def plotSol(file_name, train_sol, decoded_sol):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), subplot_kw={'projection':'3d'})
    solarray, sizenew = my_readwrite.solution_untrim(train_sol, MM - 2 * Mtrim, Mtrim)
    sol_val, umesh, vmesh = my_viz_routines.eval0k2DUV_sol(solarray[0,:], (-3, 3, -3, 3, -3, 3), (MM, MM, MM),
                                                        uval, vval, 0.0)
    axes[0].plot_surface(umesh, vmesh, sol_val, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    axes[0].set_title("Train solution")

    solarray, sizenew = my_readwrite.solution_untrim(decoded_sol, MM - 2 * Mtrim, Mtrim)
    sol_val1, umesh, vmesh = my_viz_routines.eval0k2DUV_sol(solarray[0, :], (-3, 3, -3, 3, -3, 3), (MM, MM, MM),
                                                       uval, vval, 0.0)
    axes[1].plot_surface(umesh, vmesh, sol_val1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    axes[1].set_title("Decoded solution")

    axes[2].plot_surface(umesh, vmesh, sol_val-sol_val1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    axes[2].set_title("Error")
    fig.savefig(file_name)
    plt.show()

# Main entry starting from here
settings = Utilities.LoadSettings()

if (int(settings["predict_type"]) == 0):
    sol_data_train, solsize = loadSolData()
else:
    sol_data_train, coll_data_train, solsize = loadSolData()
print(solsize)

# Load the trained model and predict decoded solutions from solutions
import tensorflow as tf
from tensorflow import keras

#autoencoder = keras.models.load_model("Model-10%-noise-HL3-CL32.hdf5")

# autoencoder = keras.models.load_model("Model-coll-HL3-CL64.hdf5")
# autoencoder = keras.models.load_model("Model-HL3-CL64.hdf5")
# autoencoder = keras.models.load_model("Model-coll-HL3-CL64.hdf5")
autoencoder = keras.models.load_model("Model-coll-HL5-CL32.hdf5")
decoded_sols = autoencoder.predict(sol_data_train)

for i in range(0, int(settings['num_test_solutions'])):
    if (int(settings["predict_type"]) == 0):
        max_sol= np.amax(np.abs(sol_data_train[i,:]))
        max_sol = np.max([max_sol, 1.0e-6])
        error = np.amax(np.absolute(sol_data_train[i,:]-decoded_sols[i,:]))/max_sol  
    else:  
        max_sol= np.amax(np.abs(coll_data_train[i,:]))
        max_sol = np.max([max_sol, 1.0e-6])
        error = np.amax(np.absolute(coll_data_train[i,:]-decoded_sols[i,:]))/max_sol
    print(f"{error:.4f}")

for i in range(0, int(settings['num_test_solutions'])):
    if (int(settings["predict_type"]) == 0):
        error = np.absolute(sol_data_train[i,:] - decoded_sols[i,:])
    else:
        error = np.absolute(coll_data_train[i,:] - decoded_sols[i,:])
    print(f"mean = {error.mean():.4f}, std = {error.std():.4f}, var = {error.var():.4f}, min = {error.min():.4f}, max = {error.max():.4f}")
    
##################### PLOTTING A FEW SOLUTIONS and THE RESULTS OF ENCODING DECODING OPERATIONimport matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

### TRIMMING PARAMETERS
MM = 41
Mtrim = 0

#### VISUALIZE THE RESULTS...
### Plot train and decoded solutions
import my_viz_routines
uval = np.arange(-3.0,2.99,0.1)
vval = np.arange(-3.0,2.99,0.1)
for i in range(0,len(sol_data_train),1):
    if (int(settings["predict_type"]) == 0):
        plotSol(f"Study solution {i+1}", sol_data_train[i], decoded_sols[i])
    else:
        plotSol(f"Study solution {i+1}", coll_data_train[i], decoded_sols[i])

