import numpy as np
import tensorflow as tf
from tensorflow import keras

import my_readwrite
import my_viz_routines
import my_utils

from tkinter import *
from tkinter import ttk
from tkinter import filedialog


###########################################################################
####  First we Set up mesh sizes and trimming: Copy and paste from the main.py
####  used to train the model above.
    ####
MM = 41
Mtrim = 5
dms=MM-2*Mtrim
####
####  END seeting up mesh sizes and trim ####################################


#########################################################################
#### Small but important: a bunch of constants to match to the
#### deterministic solver for the 0D Boltzmann equation
beta_coeff = 1.0
#### End bunch of constants for Boltzmann
#########################################################################

def ComputeSolution():
    # Get initial value
    sol, solsize = my_readwrite.my_read_solution_trim(IVVar.get(),MM, Mtrim)

    # Get  arrays of velocity points and gauss weights. these
    # points can be obtained from the DG-Boltzmann solution saves:
    filename='C:/Courses/alexdata/take3_M41/080/CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat'
    nodes_u, nodes_v, nodes_w, nodes_gwts = my_readwrite.read_nodes(filename)

    #########################################################################
    #  Prepare rad decay filter factor array
    nodes_u_trim, new_size = my_readwrite.solution_trim(nodes_u,MM,Mtrim)
    nodes_v_trim, new_size = my_readwrite.solution_trim(nodes_v,MM,Mtrim)
    nodes_w_trim, new_size = my_readwrite.solution_trim(nodes_w,MM,Mtrim)
    nodes_gwts_trim, new_size = my_readwrite.solution_trim(nodes_gwts,MM,Mtrim)

    Mat_Moments = np.zeros((new_size,5))
    Mat_Moments[:,0] = nodes_gwts_trim[0,:]
    Mat_Moments[:,1] = nodes_gwts_trim[0,:]*nodes_u_trim[0,:]
    Mat_Moments[:,2] = nodes_gwts_trim[0,:]*nodes_v_trim[0,:]
    Mat_Moments[:,3] = nodes_gwts_trim[0,:]*nodes_w_trim[0,:]
    scrp_array = (nodes_u_trim**2+nodes_v_trim**2+nodes_w_trim**2)
    Mat_Moments[:,4] = nodes_gwts_trim[0,:]*scrp_array[0,:]

    radial_values=np.sqrt(nodes_u_trim**2+nodes_v_trim**2+nodes_w_trim**2)
    radial_filter_array=my_utils.sigmoid_radial_filter(radial_values,10.0,1.7)

    Q_m, R_m = np.linalg.qr(Mat_Moments)

    ############# PREPARE SVD FILTER MATRIX
    import pickle
    #save_file = open('svd_41_7_first_101.dat', 'rb')
    save_file = open('sol_SVD_100.dat', 'rb')
    lsvect= pickle.load(save_file)
    s = pickle.load(save_file)
    svect_untrimed = pickle.load(save_file)
    save_file.close()
    #print(s[0:100])
    ### We need to trim the singular vectors 
    svect = np.zeros((100, dms**3))
    for i in range(svect_untrimed.shape[1]):
        z, solsize = my_readwrite.solution_trim(svect_untrimed.T[i:i+1,:],MM,Mtrim) 
        svect[i,:] = z
    ## all done 
    #############################
    Flt_Zero_tmp = svect - np.matmul(np.matmul(svect,Q_m),Q_m.T)
    Flt_Cons_tmp = np.matmul(np.matmul(svect,Q_m),Q_m.T)
    Flt_Zero, R_zero = np.linalg.qr(Flt_Zero_tmp.T)
    Flt_Cons, R_cons = np.linalg.qr(Flt_Cons_tmp.T)
    N_Cons=1

    ######################################################################
    ### setting up the Euler time stepping, macroparameter evaluations and
    ### solutions saves.
    init_time = 0.0       # initial time
    fin_time=0.2   # final time
    dt=0.0002      # time step
    ### some constants defining moments tracking and checkpoint saves:
    num_save_sol = 10         # how many times solution is saved
    num_eval_moments = 800    # how many times moments are tracked
    delta_t_save_sol = (fin_time-init_time)/num_save_sol
    delta_t_eval_moms = (fin_time-init_time)/num_eval_moments
    ######
    time=init_time

    ############# SAVE INITIAL DATA and compute Moments
    sol_utm, lsize = my_readwrite.solution_untrim(sol[0,:], MM - 2 * Mtrim, Mtrim)
    rec_moments = my_utils.get_moments(sol_utm,nodes_u, nodes_v, nodes_w, nodes_gwts,time)

    next_time_save_sol  = init_time + delta_t_save_sol
    next_time_eval_moms = init_time + delta_t_eval_moms
    ############ end save initial data

    ####################
    ### Splitting off the conserved part
    sol_Cons = np.matmul(np.matmul(sol,Flt_Cons[:,0:N_Cons]),Flt_Cons[:,0:N_Cons].T)
    sol_Zero = np.matmul(np.matmul(sol,Flt_Zero),Flt_Zero.T)

    ######################################################################
    #### THE MAIN LOOP
    
    count_auto_project=0
    while time < fin_time:
        ######## Record converved moments ###########4
        conserved_macroparams = my_utils.compute_conservative_moments(sol, nodes_u, nodes_v, nodes_w, nodes_gwts, MM, Mtrim)
        #conserved_macroparams = my_utils.compute_conservative_moments(sol_Cons+sol_Zero, nodes_u, nodes_v, nodes_w, nodes_gwts, MM, Mtrim)
        sol_5d = np.reshape(sol, (1, dms, dms, dms, 1))
        #sol_5d = np.reshape(sol_Cons+sol_Zero, (1, dms, dms, dms, 1))
        coll_oper_5d = learncollision.predict(sol_5d)
        coll_oper = np.reshape(coll_oper_5d, (1, dms*dms*dms))
        #coll_oper = coll_oper*(max_val-min_val)+min_val

        ###################################
        ## Apply radial zero filter
        # apply_radial_filter = False
        apply_radial_filter = radialZeroFilter.get()
        if apply_radial_filter:
            coll_oper = coll_oper * radial_filter_array

        ##################################
        ## enforce projection of the collision operator on zero mass/mom/energy space.
        # enf_projection_Zero = True
        enf_projection_Zero = enforceProjectZero.get()
        if enf_projection_Zero:
            coll_oper=np.matmul(np.matmul(coll_oper,Flt_Zero),Flt_Zero.T)
        ## end enforce projection of the collision operator on zero mass/mom/energy space
        #################################

        ##################################
        ## enforce conservation of mass on collision operator.
        # enf_conservation_mass = False
        enf_conservation_mass = enforceConvervationMass.get()
        if enf_conservation_mass:
            coll_oper, coll_op_size = my_utils.enf_conservation(coll_oper, nodes_u, nodes_v, nodes_w, nodes_gwts,MM,Mtrim)
        ## end enforce conservation of mass

        #################################
        sol[0,:] = sol[0,:] + dt*beta_coeff*coll_oper
        #sol_Zero[0, :] = sol_Zero[0, :] + dt * beta_coeff * coll_oper
        time=time+dt

        ###################################
        ## filtering using SVD of solutions
        # do_filter_using_SV = False
        do_filter_using_SV = filteringUsingSVD.get()
        if do_filter_using_SV:
            ggg=100
            sol[0,:] = np.dot(np.dot(sol[0,:],svect[0:ggg,:].T),svect[0:ggg,:])
        ## end filtering using SVD of solutions
        ####################################

        ###################################
        ## Filtering using autencoder
        # do_autoencoder_filter = False
        do_autoencoder_filter = filteringUsingAE.get()
        count_auto_project=count_auto_project+1
        #if time>next_time_eval_moms:   
        #    do_autoencoder_filter=True
        if do_autoencoder_filter and count_auto_project > 10:
            count_auto_project = 0
            sol = AE_model.predict(sol)
        ### End filtering using autoencoder
        ###################################

        ## enforce conservation of mass on solutions.
        # enf_conservation_on_solutions = False
        enf_conservation_on_solutions = enforceConservationSol.get()
        if enf_conservation_on_solutions:
            sol, sol_size = my_utils.enf_conservation_sol(sol, nodes_u, nodes_v, nodes_w, nodes_gwts, MM, Mtrim,conserved_macroparams)
        ## end enforce conservation of mass

        #############################################
        ## Now we check if it is time to record the moments
        if time>next_time_eval_moms:
            sol_utm, iizizer = my_readwrite.solution_untrim(sol[0,:],MM-2*Mtrim,Mtrim)
            #sol_utm, iizizer = my_readwrite.solution_untrim(sol_Zero[0, :]+sol_Cons[0,:], MM - 2 * Mtrim, Mtrim)
            entry_moments = my_utils.get_moments(sol_utm,nodes_u, nodes_v, nodes_w, nodes_gwts,time)
            rec_moments = np.concatenate((rec_moments, entry_moments), axis = 0)
            next_time_eval_moms =next_time_eval_moms+delta_t_eval_moms

        #############################################
        ## Add here subroutine to save solution
        ##
        if time>next_time_save_sol:
            my_readwrite.save_moments('moments.txt',rec_moments)
            next_time_save_sol=next_time_save_sol+delta_t_save_sol

    ############ ALL done. Last save and quit
    sol_utm, iisizer = my_readwrite.solution_untrim(sol[0,:], MM - 2 * Mtrim, Mtrim)
    entry_moments = my_utils.get_moments(sol_utm, nodes_u, nodes_v, nodes_w, nodes_gwts, time)
    rec_moments = np.concatenate((rec_moments, entry_moments), axis=0)
    my_readwrite.save_moments('moments.txt',rec_moments)


# Layout GUI
def SelectColOpModel():
    global learncollision
    filename = filedialog.askopenfilename(initialdir="C:/Courses/Math698A-Thesis/learn_coll_op_cnn-MAE-HL5-03_12_2021-08_46_36", 
                                        title="select a model", 
                                        filetypes=(("HDF5", "*.hdf5"),("HDF5", "*.h5")))
    modelVar.set(filename)
    learncollision = keras.models.load_model(modelVar.get())

def SelectAEModel():
    global AE_model
    filename = filedialog.askopenfilename(initialdir="C:/Courses/Math698A-Training-results", 
                                        title="select a model", 
                                        filetypes=(("HDF5", "*.hdf5"),("HDF5", "*.h5")))
    AE_modelVar.set(filename)
    AE_model = keras.models.load_model(modelVar.get())

def SelectInitialValue():
    filename = filedialog.askopenfilename(initialdir="C:/Courses/alexdata/sphomruns", 
                                        title="select an intial value file", 
                                        filetypes=(("initial data files", "*.dat"), ("all files", "*.*")))
    IVVar.set(filename)

# main entry start from here
root = Tk()
root.title("Neural Network Boltzmann Equation Solver")
f = ttk.Frame(root)
f.grid(row=0, column=0, sticky=(N,S,E,W), padx=10, pady=10)

# Select collisiton operator model
ttk.Button(f, text="Select a CNN model", command=SelectColOpModel).grid(row=0,column=0, pady=5, sticky="w")
modelVar = StringVar()
modelVar.set("C:/Courses/Math698A-Thesis/learn_coll_op_cnn-MAE-HL5-03_12_2021-08_46_36/BestLearnColOpModel.hdf5")
ttk.Label(f, textvariable=modelVar).grid(row=0, column=1, padx=5, pady=5, sticky="w")

learncollision = keras.models.load_model(modelVar.get())

# Select initial value file
ttk.Button(f, text="Select an initial value file", command=SelectInitialValue).grid(row=1,column=0, pady=5, sticky="w")
IVVar = StringVar()
IVVar.set("C:/Courses/alexdata/sphomruns/160/CollTrnDta268_2kc1su1sv1sw3NXU41MuUU41MvVU41MwWU_time0.0000000000_SltnColl.dat")
ttk.Label(f, textvariable=IVVar).grid(row=1, column=1, padx=5, pady=5, sticky="w")

# Select Auto Encoder model
ttk.Button(f, text="Select an Auto Encoder model", command=SelectAEModel).grid(row=2,column=0, pady=5, sticky="w")
AE_modelVar = StringVar()
AE_modelVar.set("C:/Courses/Math698A-Training-results/learn_sol-20%-noise-HL3-CL32-02_19_2021-12_52_38/LearnSolModel.hdf5")
ttk.Label(f, textvariable=AE_modelVar).grid(row=2, column=1, padx=5, pady=5, sticky="w")
AE_model = keras.models.load_model(AE_modelVar.get())

# GUI for Other options of computing
# Apply radial zero filter
radialZeroFilter = BooleanVar()
ttk.Checkbutton(f, text="Apply radial zero filter", variable=radialZeroFilter).grid(row=3, column=0, pady=5, sticky="w")

#  Enforce projection of the collision operator on zero mass/mom/energy space.
enforceProjectZero = BooleanVar()
ttk.Checkbutton(f, text="Enforce projection zero", variable=enforceProjectZero).grid(row=4, column=0, pady=5, sticky="w")

# Enforce conservation of mass on collision operator.
enforceConvervationMass = BooleanVar()
ttk.Checkbutton(f, text="Enforce conservation of mass ", variable=enforceConvervationMass).grid(row=5, column=0, pady=5, sticky="w")

# filtering using SVD of solutions
filteringUsingSVD = BooleanVar()
ttk.Checkbutton(f, text="Filtering using SVD", variable=filteringUsingSVD).grid(row=6, column=0, pady=5, sticky="w")

# Filtering using autencoder
filteringUsingAE = BooleanVar()
ttk.Checkbutton(f, text="Filtering using Auto Encoder", variable=filteringUsingAE).grid(row=7, column=0, pady=5, sticky="w")

# Enforce conservation of mass on solutions.
enforceConservationSol = BooleanVar()
ttk.Checkbutton(f, text="Apply conservation of mass", variable=enforceConservationSol).grid(row=8, column=0, pady=5, sticky="w")

# Button for compute solutions
ttk.Button(f, text="Compute solution", command=ComputeSolution).grid(row=9, column=0, pady=5, sticky="w")

root.mainloop()