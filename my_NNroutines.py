########################
# 08/27/2019 A. Alekseenko
# This module contains functions and
# subroutines that read and write data to hard drive
########################

############################################
# get_NNinputderTP (encoder,sol,dsol):
#
# This subroutine will use three point derivative to
# compute derivative of the neural network with respect to input
#
# encoder : is the pre-trained network, set up to convert sol in to output
# sol the state of the input at which the derivative is computed
# dsol  : step size to compute the derivative
# enc_dim dimension of the encoded output
# dim of the solution vector
##############################################

def get_NNinputderTP (encoder,sol,dsol,sol_dim,enc_dim):
    import numpy as np
    sol.shape = (1,sol_dim)
    df = np.zeros((1,sol_dim))
    gradient = np.zeros((enc_dim,sol_dim)) # set up a zero derivative

    #####
    for i in range(sol_dim):
        df[0,i] = dsol
        f1 = encoder.predict(sol-df)
        f2 = encoder.predict(sol+df)
        gradient[:,i]=(f2-f1)/2/dsol
        df[0,i] = 0.0
    return gradient

###########################################
# get_NNinputderTP_ilist (encoder,sol,dsol):
# This is a copy of the above subroutine, except not all
# components of the gradient arte computed.
# The indices of components that are computed are provided in list ilist
#
# This subroutine will use three point derivative to
# compute derivative of the neural network with respect to input
#
# encoder : is the pre-trained network, set up to convert sol in to output
# sol the state of the input at which the derivative is computed
# dsol  : step size to compute the derivative
# enc_dim dimension of the encoded output
# dim of the solution vector
##############################################

def get_NNinputderTP_ilist (encoder,sol,dsol,sol_dim,enc_dim,ilist):
    import numpy as np
    sol.shape = (1,sol_dim)
    df = np.zeros((1,sol_dim))
    gradient = np.zeros((enc_dim,sol_dim)) # set up a zero derivative

    #####
    for i in ilist:
        df[0,i] = dsol
        f1 = encoder.predict(sol-df)
        f2 = encoder.predict(sol+df)
        gradient[:,i]=(f2-f1)/2/dsol
        df[0,i] = 0.0
    return gradient



#################################################
#
#
# This subroutine computes the derivative of the NN adaptively
#  It first evaluates it at some step size sdol then halves the stepsize untill convergence is
#  established. Indicator is used to estimate truncation error.
#
#  For second order method of differentialtion, the indicator is
# e=(D(h/2)-D(h))*4/3
# Richardson extrapolation is
#I=(4*D(h/2) -D(h))/3
#
# encoder : is the pre-trained network, set up to convert sol in to output
# sol the state of the input at which the derivative is computed
# dsol  : step size to compute the derivative
# enc_dim dimension of the encoded output
# dim of the solution vector
#
####################################################

def eval_NNderivative_adaptive(encoder,sol,dsol,sol_dim,enc_dim):
    import numpy as np
    eps = 1.0e-4 # threshhold to stop refinement
    err=1000.0
    df=dsol
    grad1 = get_NNinputderTP(encoder, sol, df, sol_dim, enc_dim)
    grad2 = get_NNinputderTP(encoder, sol, df / 2.0, sol_dim, enc_dim)
    err = (np.absolute(grad2 - grad1)) / max(np.amax(np.absolute(grad2)), 1.0e-6)
    refine_list=[]
    for i in range(sol_dim):
        if np.amax(err[:,i]) > eps:
            refine_list.append(i)
    # use the grad computed with a smaller time step
    grad1 = grad2
    df =df/2.0
    i = 0
    ## adaptive refinement in df on selected columents of gradient
    while (len(refine_list)>0 and (i < 10)):
        grad2 = get_NNinputderTP_ilist (encoder,sol,df/2.0,sol_dim,enc_dim,refine_list)
        err = (np.absolute(grad2-grad1))/max(np.amax(np.absolute(grad1)),1.0e-6)
        i = i+1
        # update the list for mesh step refinement
        ierr = 1.0e-9
        refine_list1 = []
        for j in refine_list:
            grad1[:, j] = grad2[:, j]
            if np.amax(err[:, j]) > eps:
                refine_list1.append(j)
                ierr=max(ierr,np.amax(err[:,j]))
        # use the grad computed with a smaller time step
        df=df/2.0
        refine_list=refine_list1
        print("i is ", i, "err is", ierr)
    return grad1