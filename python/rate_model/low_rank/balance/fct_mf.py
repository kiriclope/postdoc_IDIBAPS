import numpy as np
import scipy
from scipy.optimize import fsolve
from functools import partial

import fct_facilities as fac
from fct_integrals import *

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Solve mean-field equations

### Zero-mean solutions, corresponding to the central solution, solved through iteration

def SolveStaticZero (y0, g, VecPar, tolerance = 1e-8, backwards = 1):

    # The variable y contains the mean-field variables mu and delta0
    # The variable backwards can be set to (-1) to invert the flow of iteration and reach unstable solutions

    again = 1
    y = np.array(y0)
    y_new = np.ones(2)
    eps = 0.2

    while (again==1):

        # Take a step

        new0 = np.sqrt(K) * ( gv.I0 - gv.J0 * intPhi(y[0],y[1]) ) # net input
        new1 = gv.J0 * gv.J0 * intPhiSq(y[0], y[1]) # total variance
        
        y_new[0] = (1-backwards*eps)*y[0] + eps*backwards*new0
        y_new[1] = (1-eps)*y[1] + eps*new1

        # Stop if the variables converge to a number, or zero
        # If it becomes nan, or explodes

        if ( np.fabs(y[1]-y_new[1]) < tolerance*np.fabs(y[1]) ):
            again = 0

        if ( np.fabs(y[1]-y_new[1]) < tolerance ):
            again = 0

        if np.isnan(y_new[0]) == True:
            again = 0
            y_new = [0,0]

        if( np.fabs(y[0])> 1/tolerance  ):
            again = 0
            y_new = [0,0]
    
        y[0] = y_new[0]
        y[1] = y_new[1]

    return y_new

### Non-zero solutions, solved through iteration
def SolveStaticNonZero (y0, g, VecPar, tolerance = 1e-8, backwards = 1):

    # The variable y contains the mean-field variables mu, delta0 and deltainf
    # The variable backwards can be set to (-1) to invert the flow of iteration and reach unstable solutions

    again = 1
    y = np.array(y0)
    y_new = np.ones(3)
    eps = .2

    while (again==1):

        # Take a step

        new0 = np.sqrt(K) * ( gv.I0 - gv.J0 * intPhi(y[0],y[1]) ) # net input 
        new1 = gv.J0 * gv.J0 * intPhiSq(y[0], y[1]) # total variance 

        new0 = Mm*Mn*Phi(y[0], y[1]) 
        new1 = np.sqrt( np.max ( [0, y[2]**2 +2* (g*g* ( PrimSq(y[0], y[1]) - IntPrimPrim(y[0], y[1], y[2]))  + (Mn**2 * Sim**2 * Phi(y[0], y[1])**2 ) * (y[1] - y[2]) ) ] ) )
        new2 = g*g*IntPhiPhi(y[0], y[1], y[2]) + Mn**2 * Sim**2 * Phi(y[0], y[1])**2

        y_new[0] = (1-backwards*eps)*y[0] + backwards*eps*new0
        y_new[1] = (1-eps)*y[1] + eps*new1
        y_new[2] = (1-eps)*y[2] + eps*new2

        # Stop if the variables converge to a number, or zero
        # If it becomes nan, or explodes
        # If delta_inf becomes larger than delta0

        if( np.fabs(y[1]-y_new[1]) < np.fabs(y[1])*tolerance and np.fabs(y[0]-y_new[0]) < np.fabs(y[0])*tolerance):
            again = 0

        if( np.fabs(y[1]-y_new[1]) < tolerance and np.fabs(y_new[0]) < tolerance):
            again = 0

        if( np.fabs(y[0])> 1/tolerance  ):
            again = 0

        if np.isnan(y_new[0]) == True:
            again = 0

        if (y_new[2]>y_new[1]):
            again = 0
    
        y[0] = y_new[0]
        y[1] = np.fabs(y_new[1])
        y[2] = np.fabs(y_new[2])

    return y_new
