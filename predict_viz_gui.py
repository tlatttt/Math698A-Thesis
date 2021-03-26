##########################################################################################################
# Math 689: Thesis
# Student: Thomas V Nguyen
# Advisor: Prof. Alekseenko
#
# This module is a GUI (graphical user interface) for a user to load a trained neural network model
# and test it with the random or selected data
# 
##########################################################################################################

from tkinter import *
from tkinter import ttk
from tkinter import filedialog

import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import my_readwrite
import my_viz_routines
import Utilities

# define global variables
sol_data_train = []
coll_data_train = []
decoded_sols = []
solsize = 0

def LoadSolutions():
    names = my_readwrite.my_get_soltn_file_names_time (datapath.get(), cutoff_time.get())
    num_sols = len(names)  # total available solutions
    test_sols_names = []   #prepare an empty array that will keep names of the solutions to study
    for x in range(numTestSols.get()):
        ii=random.randint(0,num_sols-1)
        test_sols_names.append(names[ii])
    MM = int(MMVar.get())
    Mtrim = int(MtrimVar.get())
    sol_data_train, coll_data_train, solsize = my_readwrite.my_read_sol_coll_trim(test_sols_names[0],MM, Mtrim)  # this is the first file
    zz = len(test_sols_names)
    for i in range(1,zz):
        solarry, collarry, solsize1 = my_readwrite.my_read_sol_coll_trim(test_sols_names[i], MM, Mtrim)
        assert solsize1==solsize  # a check that all arrays have the same size -- should be true, but still..
        sol_data_train = np.concatenate((sol_data_train, solarry), axis = 0)
        coll_data_train = np.concatenate((coll_data_train, collarry), axis=0) 
    
    return test_sols_names, sol_data_train, coll_data_train, solsize

def SelectModel():
    global model
    filename = filedialog.askopenfilename(initialdir="C:/Courses/Math698A-Thesis", 
                                        title="select a model", 
                                        filetypes=(("HDF5", "*.hdf5"),("HDF5", "*.h5")))
    modelVar.set(filename)
    model = keras.models.load_model(modelVar.get())


def SelectDataFolder():
    dir = filedialog.askdirectory(initialdir="C:/Courses/alexdata/sphomruns")
    datapath.set(dir)

def RandomPredicts():
    global sol_data_train, coll_data_train, decoded_sols, solsize
    solfiles, sol_data_train, coll_data_train, solsize = LoadSolutions()
    solfilesVar.set(solfiles)
    if (learnType.get() == "ls"):
        decoded_sols =model.predict(sol_data_train)
        computeErrors(sol_data_train, decoded_sols)
    else:
        if (learnType.get() == "lcocnn"):
            dms = int(np.cbrt(solsize))
            sol_data_train = np.reshape(sol_data_train, (len(sol_data_train), dms,dms,dms,1))
        decoded_sols =model.predict(sol_data_train)
        computeErrors(coll_data_train, decoded_sols)

    solsizeVar.set("Solution size: " + str(solsize))

def computeErrors(sol_data_train, decoded_sols):
    errors.clear()
    staErrors.clear()
    for i in range(0, int(numTestSols.get())):
        max_sol= np.amax(np.abs(sol_data_train[i,:]))
        max_sol = np.max([max_sol, 1.0e-6])
        if (learnType.get() == "ls"):
            error = np.amax(np.absolute(sol_data_train[i,:]-decoded_sols[i,:]))/max_sol
        else:
            error = np.amax(np.absolute(coll_data_train[i,:]-decoded_sols[i,:]))/max_sol
        errors.append(error)
        e = np.absolute(sol_data_train[i,:]-decoded_sols[i,:])
        staErrors.append(f"mean = {e.mean():.4f}, std = {e.std():.4f}, var = {e.var():.4f}, min = {e.min():.4f}, max = {e.max():.4f}")
    errorListVar.set(errors)
    staErrorListVar.set(staErrors)

def plotOnePredictedSol(train_sol, decoded_sol):
    MM = int(MMVar.get())
    Mtrim = int(MtrimVar.get())
    uval = np.arange(-3.0,2.99,0.1)
    vval = np.arange(-3.0,2.99,0.1)

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
    plt.show()

def plotPredictedSol(*args):
    idx = solList.curselection()
    i = int(idx[0])
    if (learnType.get() == "ls"):
        train_sol = sol_data_train[i]
    else:
        train_sol = coll_data_train[i]
    decoded_sol = decoded_sols[i]
    plotOnePredictedSol(train_sol, decoded_sol)

def PlotOnePredictedSol():
    solfile = filedialog.askopenfilename(initialdir="C:/Courses/alexdata/sphomruns", 
                                title="Select a solution file", 
                                filetypes=(("solution file", "*.dat"),("all", "*.*")))
    if (len(solfile) == 0):
        return

    MM = int(MMVar.get())
    Mtrim = int(MtrimVar.get())
    sol_train, coll_train, solsize = my_readwrite.my_read_sol_coll_trim(solfile,MM, Mtrim)  # this is the first file
    if (learnType.get() == "lcocnn"):
        dms = int(np.cbrt(solsize))
        sol_train = np.reshape(sol_train, (len(sol_train), dms,dms,dms,1))
    decoded_sol = model.predict(sol_train)
    if (learnType.get() == "ls"):
        plotOnePredictedSol(sol_train[0],decoded_sol[0])
    else:
        plotOnePredictedSol(coll_train[0],decoded_sol[0])    


# Layout GUI
root = Tk()
root.title("Predict and Visualiztion")
f = ttk.Frame(root)
f.grid(row=0, column=0, sticky=(N,S,E,W), padx=10, pady=10)

# Choose learning type: learing solution or learning collision operator
learnType = StringVar()
learnType.set("ls")
ttk.Radiobutton(f, text="Learn Solutions", variable=learnType, value="ls").grid(row=0, column=0, pady=5)
ttk.Radiobutton(f, text="Learn Collision Operator", variable=learnType, value="lco").grid(row=0,column=1,pady=5)
ttk.Radiobutton(f, text="Learn Collision Operator CNN", variable=learnType, value="lcocnn").grid(row=0,column=2,pady=5)

# Trimming parameters and solution info
MMVar = IntVar()
MMVar.set(41)
MtrimVar = IntVar()
MtrimVar.set(5)
cutoff_time = DoubleVar()
cutoff_time.set(0.30)
ttk.Label(f, text="Trim parameters: ").grid(row=1, column=0, pady=5, sticky="w")
ttk.Label(f, text="MM:").grid(row=1,column=1,sticky="e",padx=5)
MM_spin = ttk.Spinbox(f, from_=20, to=41, width=5, textvariable=MMVar).grid(row=1, column=2, sticky="w")
ttk.Label(f, text="Mtrim:").grid(row=1,column=3,sticky="e",padx=5)
Mtrim_spin = ttk.Spinbox(f, from_=3, to=7, width=5, textvariable=MtrimVar).grid(row=1, column=4, sticky="w")
ttk.Label(f, text="Cutoff time:").grid(row=1,column=5,padx=5)
cutofftime_spin = ttk.Spinbox(f, from_=0.10, to=0.30,width=5, increment=0.01,textvariable=cutoff_time).grid(row=1,column=6, sticky="w")
solsizeVar = StringVar()
solsizeVar.set("Solution size:")
ttk.Label(f, textvariable=solsizeVar).grid(row=1, column=7, sticky="w")

# Select model file
modelSelBtn = ttk.Button(f, text="Select a model", command=SelectModel).grid(row=2,column=0, pady=5, sticky="w")
modelVar = StringVar()
modelVar.set("C:/Courses/Math698A-Thesis/LearnSolModel.hdf5")
model_path = ttk.Label(f, textvariable=modelVar).grid(row=2, column=1, padx=5, pady=5, sticky="w")

# Select data folder
dataSelBtn = ttk.Button(f, text="Select data folder", command=SelectDataFolder).grid(row=3,column=0, pady=5, sticky="w")
datapath = StringVar()
datapath.set("C:/Courses/alexdata/sphomruns")
dataPathLbl = ttk.Label(f, textvariable=datapath).grid(row=3,column=1,padx=5,pady=5, sticky="w")

# Pick number of data files to predict and visualize
numTestSols = IntVar()
numTestSols.set(20)
ttk.Label(f, text="Nmber of test solutions:").grid(row=4, column=0, pady=5, sticky="w")
ttk.Spinbox(f, from_=1, to=100, width=5,textvariable=numTestSols).grid(row=4, column=1, sticky="w", padx=5)
ttk.Button(f, text="Random Predicts", command=RandomPredicts).grid(row=4, column=2, padx=5)
ttk.Button(f, text="Predict one solution", command=PlotOnePredictedSol).grid(row=4, column=3, padx=5)

# Listbox containing the solution data path
#solfiles, sol_data_train, solsize = LoadSolutions()
solsizeVar.set("Solution size: ?")
solfilesVar = StringVar()
solfilesVar.set("")
ttk.Label(f, text="predicted solutions: click an item to visualize").grid(row=5, column=0, columnspan=5, sticky="w", pady=5)
solList = Listbox(f, width=140, listvariable=solfilesVar)
sb = ttk.Scrollbar(f, orient=VERTICAL, command=solList.yview)
sb.grid(row=6,column=5, sticky=(W,N,S))
solList.configure(yscrollcommand=sb.set)
solList.grid(row=6, column=0, columnspan=5, sticky=(W,E),pady=5)
solList.bind('<<ListboxSelect>>', plotPredictedSol)

# Error list
ttk.Label(f, text="errors").grid(row=5, column=6, pady=5)
errors = []
errorListVar = StringVar()
errorListVar.set(errors)
errorList = Listbox(f, listvariable=errorListVar)
sb2 = ttk.Scrollbar(f, orient=VERTICAL, command=errorList.yview)
errorList.configure(yscrollcommand=sb2.set)
errorList.grid(row=6,column=6, sticky=(W,E),pady=5)
sb2.grid(row=6,column=7, sticky=(W,N,S))

# List for statistical errors
ttk.Label(f, text="Statistical errors").grid(row=7, column=0, sticky="W", pady=5)
staErrors = []
staErrorListVar = StringVar()
staErrorListVar.set(errors)
staErrorList = Listbox(f, listvariable=staErrorListVar)
sb3 = ttk.Scrollbar(f, orient=VERTICAL, command=staErrorList.yview)
staErrorList.configure(yscrollcommand=sb3.set)
staErrorList.grid(row=8,column=0, columnspan=5,sticky=(W,E),pady=5)
sb3.grid(row=8, column=5, sticky=(W,N,S))

root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)
f.rowconfigure(0, weight=1)
f.rowconfigure(1, weight=1)
f.rowconfigure(2, weight=1)
f.rowconfigure(3, weight=1)
f.rowconfigure(4, weight=1)
f.rowconfigure(5, weight=1)
f.rowconfigure(6, weight=1)
f.rowconfigure(7, weight=1)
f.rowconfigure(8, weight=1)

f.columnconfigure(0, weight=1)
f.columnconfigure(1, weight=1)
f.columnconfigure(2, weight=1)
f.columnconfigure(3, weight=1)
f.columnconfigure(4, weight=1)
f.columnconfigure(5, weight=1)
f.columnconfigure(6, weight=1)

# Load model and do predict
model = keras.models.load_model(modelVar.get())
#decoded_sols = model.predict(sol_data_train)

# Compute errors
#computeErrors(sol_data_train, decoded_sols)

root.mainloop()