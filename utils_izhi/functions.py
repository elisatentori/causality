import numpy as np

import random

import seaborn as sns

import time

import itertools
import os

from timeit import default_timer as timer


#=================================================================================================#

fs = 1000
dt = 1


################################################################################
#                               Izhikevich model                               #
################################################################################    

def izhikevich_static_delays(net, izhi_exc, izhi_inh, I_limit, I_noise_exc, I_noise_inh, path_data, path_params, path_connectivity: str = None):

    # - - - - - - - -     Neuron types
    ntypes = net['ntypes']            # (N,) boolean array with the N neurons types (exc./inh.)
    ns     = ntypes.shape[0]          # number N of neurons
    n_exc_links = np.shape(np.nonzero(net['weights'][net['weights']>0]))[1]
    ns_range = np.arange(ns)
    ones_mat = np.ones((ns,ns),dtype=int)
    # - - - - - - - -     Simulation details
    runtime = net['runtime']
    deltat  = net['deltat']
    # - - - - - - - -     Neuron parameters
    nrands  = net['nrands']            # (N,) random N numbers
    if path_connectivity:
        a,b,c,d = np.loadtxt(path_connectivity+'Parameters/model_parameters.txt', unpack=True, usecols=(0,1,2,3))
    else:
        a = ntypes*izhi_exc['a'] + (1-ntypes)*(izhi_inh['a']+izhi_inh['a_ran']*nrands)    # (N,)
        b = ntypes*izhi_exc['b'] + (1-ntypes)*(izhi_inh['b']+izhi_inh['b_ran']*nrands)    # (N,)
        nrsquared = nrands*nrands                                                         # (N,)
        c = ntypes*(izhi_exc['c']+izhi_exc['c_ran']*nrsquared) + (1-ntypes)*izhi_inh['c'] # (N,)
        d = ntypes*(izhi_exc['d']+izhi_exc['d_ran']*nrsquared) + (1-ntypes)*izhi_inh['d'] # (N,)
    parameters = np.column_stack((a, b, c, d))
    np.savetxt(path_params+"model_parameters.txt", parameters, delimiter=' ')
    print('Model params shape:  a',np.shape(a),'  b',np.shape(b),'  c',np.shape(c),'  d',np.shape(d))
    # - - - - - - - -     Connectivity matrix    (receiver,sender)
    S = np.copy(net['weights'])     # (N, N)
    S = S.T 
    print('Connectivity matrix shape:  ',np.shape(S))  
    W_max = 10#np.max(np.max(net['weights']))
    # - - - - - - - -     Delays
    D = net['delays']
    D = D.T
    # - - - - - - - -     Initial conditions
    v = -65*np.ones((ns), dtype=np.float64);   # Initial values of v
    u = np.multiply(b,v, dtype=np.float64);    # Initial values of u
    print('Potential arrays shape:  v',np.shape(v),'  u',np.shape(u))
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    #                               Izhikevich simulation                               #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    
    # - - - - - - - -     Connectivity matrix    (receiver,sender)
    S = np.copy(net['weights']).T     # (N, N)
    print('Connectivity matrix shape:  ',np.shape(S))
    tstart = timer()
    firings=np.zeros((net['neurons'],runtime));
  
    for t in range(21):
        # - - - - - - - - -     firings update
        fired=np.where(v>=30)[0];    # indices of spikes 
        bools=np.zeros((ns),dtype=bool)
        bools[fired]=True
        firings[:,t]=bools
        # - - - - - - - - -     noise currents
        I=np.zeros(ns)
        idxs_noise=np.random.choice(ns,I_limit)
        which=ntypes[idxs_noise]
        I[idxs_noise[which==True]]  = np.abs(np.random.normal(I_noise_exc,2,size=np.sum(which==True)))
        I[idxs_noise[which==False]] = np.abs(np.random.normal(I_noise_inh,2,size=np.sum(which==False)))
        # - - - - - - - - -     if some neuron fired sum to I_noise also the current 
        #                       sender_that_have_just_spiked–>
        if len(fired)!=0:    
            v[fired]=c[fired]
            u[fired]=np.add(u[fired],d[fired])
            I=I+np.sum(S[:,fired],axis=1)   # per ogni riga (receiver), sommi i posti indicati da fired
        v = v + deltat/2 * ( 0.04 * v*v + 5*v + 140 - u + I )
        v = np.clip(v, -100, 100)
        v = v + deltat/2 * ( 0.04 * v*v + 5*v + 140 - u + I )
        v = np.clip(v, -100, 100)
        u = u + deltat*(a*(b*v - u ))

    for t in range(21,runtime):
        if t%10000==0:
            print(int(t/1000),' sec')
        # - - - - - - - - -     firings update
        fired=np.where(v>=30)[0];    # indices of spikes 
        bools=np.zeros((ns),dtype=bool)
        bools[fired]=True
        firings[:,t]=bools
        # - - - - - - - - -     noise currents
        I=np.zeros(ns)
        idxs_noise=np.random.choice(ns,I_limit)
        which=ntypes[idxs_noise]
        I[idxs_noise[which==True]]  = np.abs(np.random.normal(I_noise_exc,2,size=np.sum(which==True)))
        I[idxs_noise[which==False]] = np.abs(np.random.normal(I_noise_inh,2,size=np.sum(which==False)))
        idx_D=ones_mat*t-D
        past_fired = firings[ns_range[None,:], idx_D]
        if len(fired)!=0:
            v[fired]=c[fired]
            u[fired]=np.add(u[fired],d[fired])
        I=I+np.sum((past_fired*net['weights'].T),axis=1)   # per ogni riga (receiver), sommi i posti indicati da fired
        v = v + deltat/2 * ( 0.04 * v*v + 5*v + 140 - u + I )
        v = np.clip(v, -100, 100)
        v = v + deltat/2 * ( 0.04 * v*v + 5*v + 140 - u + I )
        v = np.clip(v, -100, 100)
        u = u + deltat*(a*(b*v - u ))
    print(f"Simulation took {timer() - tstart} seconds.")

    rows, cols = np.nonzero(firings)
    np.savetxt(path_data+'spikeTimes.txt',list(zip(rows,cols)))

    return rows, cols, a, b, c, d

    
    
def prepare_numba_parameters(net, path_params, path_connectivity : str = None):
    
    # - - - - - - - -     Neuron types
    ntypes = net['ntypes']            # (N,) boolean array with the N neurons types (exc./inh.)
    ns     = ntypes.shape[0]          # number N of neurons
    n_exc_links = np.shape(np.nonzero(net['weights'][net['weights']>0]))[1]
    ns_range = np.arange(ns)
    ones_mat = np.ones((ns,ns),dtype=int)
    # - - - - - - - -     Simulation details
    runtime = net['runtime']
    deltat  = net['deltat']
    # - - - - - - - -     Neuron parameters
    nrands  = net['nrands']            # (N,) random N numbers
    if path_connectivity:
        a,b,c,d = np.loadtxt(path_connectivity+'Parameters/model_parameters.txt', unpack=True, usecols=(0,1,2,3))
    else:
        a = ntypes*izhi_exc['a'] + (1-ntypes)*(izhi_inh['a']+izhi_inh['a_ran']*nrands)    # (N,)
        b = ntypes*izhi_exc['b'] + (1-ntypes)*(izhi_inh['b']+izhi_inh['b_ran']*nrands)    # (N,)
        nrsquared = nrands*nrands                                                         # (N,)
        c = ntypes*(izhi_exc['c']+izhi_exc['c_ran']*nrsquared) + (1-ntypes)*izhi_inh['c'] # (N,)
        d = ntypes*(izhi_exc['d']+izhi_exc['d_ran']*nrsquared) + (1-ntypes)*izhi_inh['d'] # (N,)
    parameters = np.column_stack((a, b, c, d))
    np.savetxt(path_params+"model_parameters.txt", parameters, delimiter=' ')
    print('Model params shape:  a',np.shape(a),'  b',np.shape(b),'  c',np.shape(c),'  d',np.shape(d))
    # - - - - - - - -     Connectivity matrix    (receiver,sender)
    S = np.copy(net['weights'])     # (N, N)
    S = S.T 
    print('Connectivity matrix shape:  ',np.shape(S))  
    W_max = 10#np.max(np.max(net['weights']))
    # - - - - - - - -     Delays
    D = net['delays']
    D = D.T
    
    return ns, a, b, c, d, S, D, ntypes
    
    
    
from numba import njit, prange
from timeit import default_timer as timer

# Define a Numba-compiled version of the Izhikevich update loop
@njit
def static_izhikevich_numba(runtime, deltat, ns, a, b, c, d, S, D, ntypes, I_limit, I_noise_exc, I_noise_inh, nrands):
    
    v = -65 * np.ones(ns)
    u = b * v
    firings = np.zeros((ns, runtime), dtype=np.bool_)

    ones_mat = np.ones((ns, ns), dtype=np.int32)
    ns_range = np.arange(ns)

    for t in range(runtime):
        fired = np.where(v >= 30)[0]
        if fired.size > 0:
            firings[fired, t] = True
            v[fired] = c[fired]
            u[fired] += d[fired]

        I = np.zeros(ns)
        idxs_noise = np.random.choice(ns, I_limit, replace=False)
        for idx in idxs_noise:
            if ntypes[idx]:
                I[idx] = abs(np.random.normal(I_noise_exc, 2))
            else:
                I[idx] = abs(np.random.normal(I_noise_inh, 2))

        if t >= np.max(D):
            idx_D = ones_mat * t - D
            past_fired = np.zeros((ns, ns), dtype=np.float64)
            for i in range(ns):
                for j in range(ns):
                    past_fired[i, j] = firings[j, idx_D[i, j]]
            I += np.sum(past_fired * S.T, axis=1)
        elif fired.size > 0:
            I += np.sum(S[:, fired], axis=1)

        # Two half-step Euler integrations
        v += deltat / 2 * (0.04 * v * v + 5 * v + 140 - u + I)
        v = np.clip(v, -100, 100)
        v += deltat / 2 * (0.04 * v * v + 5 * v + 140 - u + I)
        v = np.clip(v, -100, 100)
        u += deltat * (a * (b * v - u))

    return firings, v, u







