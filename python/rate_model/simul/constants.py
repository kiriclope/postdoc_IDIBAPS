import numpy as np

global folder
folder = 'test'

global filename
filename = 'inputs.dat'

global IF_LOW_RANK
IF_LOW_RANK = 1

global MEAN_XI
MEAN_XI = 0

global VAR_XI
VAR_XI = 0.25 

def init_param():
    
    global n_pop
    n_pop = 1
    
    global n_neurons
    n_neurons = 1
    
    global K
    K = 500

    global m0
    m0 = .01
    
    global path
    path = '' 

    global ext_inputs
    ext_inputs = []
    
    global J
    J = []
    
    global Tsyn
    Tsyn = []

    print("reading parameters from:")
    file_name = "/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/cpp/rate_model/parameters/%dpop/%s.txt" % (n_pop, folder)
    print(file_name)

    i=0
    with open(file_name, 'r') as file:  # Open file for read
        for line in file:           # Read line-by-line
            line = line.strip().split()  # Strip the leading/trailing whitespaces and newline
            line.pop(0)
            if i==0:
                ext_inputs = np.asarray([float(j) for j in line])
                # print(ext_inputs)
            if i==1:
                J = np.asarray([float(j) for j in line])
                J = J.reshape(n_pop, n_pop)
                # print(J)
            if i==2:
                Tsyn = np.asarray([float(j) for j in line])
                Tsyn = Tsyn.reshape(n_pop, n_pop)
                # print(Tsyn)
            i=i+1

    path = '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/cpp/rate_model/simulations/%dpop/%s/N%d/K%d/' % (n_pop, folder, n_neurons, K)
    if(IF_LOW_RANK):
        path += 'low_rank/xi_%.2f_mean_%.2f_var/' % (MEAN_XI,VAR_XI)
    print('reading simulations data from:')
    print(path)
