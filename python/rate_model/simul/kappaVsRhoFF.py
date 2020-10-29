import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

import constants as gv
importlib.reload(sys.modules['constants'])

gv.J0 = 0.2 
gv.MEAN_XI = -0.0 
gv.VAR_XI = 5.0 

gv.IF_FF = 1 
gv.MEAN_FF = 0.0 
gv.VAR_FF = 0.1 
gv.VAR_ORTHO = 0.0 

gv.IF_TRIALS = 1 

for gv.TRIAL_ID in np.arange(0,11,1): 
    var = [] 
    kappas = [] 
    
    for gv.RHO_FF_XI in np.arange(0,1.05,.05): 
        gv.init_param() 
        
        try : 
            overlap = np.loadtxt(gv.path + 'overlap.dat') ; 
                
            kappa = np.mean(np.delete(overlap,[0],axis=1),axis=0) ; 
            kappas.append(kappa[0])
            var.append(1-gv.RHO_FF_XI)
                
        except :
            pass
    
    print(var)
    print(kappas)

    fig = plt.figure('kappa Vs var ortho')
    plt.plot(var,kappas,'o')

    # kappa_pos = [abs(i) for i in kappas]
    # kappa_neg = [-abs(i) for i in kappas]

    # plt.plot(J0,kappa_pos,'-')
    # plt.plot(J0,kappa_neg,'-')

    plt.xlabel('$1-\\rho$')
    plt.ylabel('$\kappa$')


plt.show()
plt.ylim([-1,1])
