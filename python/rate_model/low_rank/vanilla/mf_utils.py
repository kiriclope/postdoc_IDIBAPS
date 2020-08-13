import sys, importlib 
import numpy as np 
import scipy.special as special 

import constants as gv

def print_globals():
    print('I0', gv.I0, 'J0', gv.J0, 'J1', gv.J1, 'sigma_1', gv.sigma_1)
    print('TOL', gv.TOL, 'MAX_ITER', gv.MAX_ITER)

def phi(x): # normal distribution
    return np.exp(-0.5*x**2) / np.sqrt(2*np.pi)

def Phi(x): # CDF of the normal distribution
    return 0.5 *(1.0 + special.erf(x/np.sqrt(2)) )

def mean_rate(u, sigma):  
    return Phi( u/np.sqrt(1.0 + sigma) ) 

def variance(u, sigma):
    return np.maximum(0, gv.J0**2 * ( mean_rate(u, sigma) - 2 * special.owens_t(u/np.sqrt(1.0 + sigma), 1/np.sqrt(1.0 + 2*sigma) ) ) )

def overlap(u, sigma, kappa):
    return gv.J1*mean_rate(u,sigma)+ kappa*gv.sigma_1/np.sqrt(1+sigma)*phi(u/np.sqrt(1+sigma))

def branch_kappa_0_eqs(x):
    
    u= x[0]
    sigma = abs(x[1])

    mean_eq = u/np.sqrt(gv.K) - ( gv.I0 + gv.J0 * mean_rate(u, sigma) )  
    var_eq = sigma - variance(u, sigma)

    eqs = np.array([mean_eq, var_eq])
    return eqs.flatten()
