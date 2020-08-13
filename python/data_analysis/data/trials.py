from .std_lib import *
from . import global_vars as gv

def which_trials(y_labels):
    y_trials = []
    
    if 'ND' in gv.trial:
        # y_trials = np.argwhere( (y_labels[4]==0) & (y_labels[8]==0) & ( (y_labels[2]==1) | (y_labels[2]==4) ) ).flatten()
        y_trials = np.argwhere( (y_labels[4]==0) & (y_labels[8]==0) ).flatten()
        if 'S1' in gv.trial:
            y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[4]==0) & (y_labels[8]==0) ).flatten()
        elif 'S2' in gv.trial:
            y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[4]==0) & (y_labels[8]==0) ).flatten()

    elif 'D1' in gv.trial:
        y_trials = np.argwhere((y_labels[4]==13) & (y_labels[8]==0) ).flatten() 
        if 'S1' in gv.trial:
            y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[4]==13) & (y_labels[8]==0) ).flatten()
        elif 'S2' in gv.trial:
            y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[4]==13) & (y_labels[8]==0) ).flatten()
            
    elif 'D2' in gv.trial:
        y_trials = np.argwhere((y_labels[4]==14) & (y_labels[8]==0)).flatten()
        if 'S1' in gv.trial:
            y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[4]==14) & (y_labels[8]==0) ).flatten()
        elif 'S2' in gv.trial:
            y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[4]==14) & (y_labels[8]==0) ).flatten()

    elif 'paired' in gv.trial :
        y_trials = np.argwhere( (y_labels[4]==0) & (y_labels[8]==0) &
                                          ((y_labels[0]==17) & (y_labels[1]==11)) |
                                          ((y_labels[0]==18) & (y_labels[1]==12)) ).flatten()
    elif 'unpaired' in gv.trial :
        y_trials = np.argwhere( (y_labels[4]==0) & (y_labels[8]==0) &
                                          ((y_labels[0]==17) & (y_labels[1]==12)) | 
                                          ((y_labels[0]==18) & (y_labels[1]==11)) ).flatten()         
        
    return y_trials

def get_S1_S2_trials(X_data, y_labels):
    y_S1_trials = which_trials(y_labels, 'S1' + gv.trial)
    y_S2_trials = which_trials(y_labels, 'S2' + gv.trial)

    X_S1_trials = X_data[y_S1_trials]
    X_S2_trials = X_data[y_S2_trials]

    return y_S1_trials, X_S1_trials, y_S2_trials, X_S2_trials

def get_X_y_ED_LD(X_S1_trials, X_S2_trials):

    X_S1_ED = np.mean(X_S1_trials[:,:,gv.bins_ED],axis=2)
    X_S1_LD = np.mean(X_S1_trials[:,:,gv.bins_LD],axis=2)

    X_S2_ED = np.mean(X_S2_trials[:,:,gv.bins_ED],axis=2)
    X_S2_LD = np.mean(X_S2_trials[:,:,gv.bins_LD],axis=2)

    X_S1_ED_LD = np.asarray([X_S1_ED, X_S1_LD])
    X_S2_ED_LD = np.asarray([X_S2_ED, X_S2_LD])

    X_ED_LD = np.concatenate([X_S1_ED_LD, X_S2_ED_LD],axis=1)

    y_S1 = np.repeat(0,int(X_S1_trials.shape[0]))
    y_S2 = np.repeat(1, int(X_S2_trials.shape[0]))

    y_ED_LD = np.concatenate((y_S1, y_S2))
    
    return X_ED_LD, y_ED_LD

def bin_data(data, bin_step, bin_size):
    # bin_step number of pts btw bins, bin_size number of size in each bin
    bin_array = [np.mean(np.take(data,np.arange(int(i*bin_step),int(i*bin_step+bin_size)), axis=2), axis=2) for i in np.arange(data.shape[2]//bin_step-1)]
    bin_array = np.array(bin_array)
    bin_array = np.rollaxis(bin_array,0,3)
    return bin_array

def get_X_y_trials(X_data, y_labels):
    y_S1_trials, X_S1_trials, y_S2_trials, X_S2_trials = get_S1_S2_trials(X_data, y_labels)

    X_S1 = bin_data(X_S1_trials, gv.n_bin, gv.n_bin)
    X_S2 = bin_data(X_S2_trials, gv.n_bin, gv.n_bin)

    X_trials = np.concatenate([X_S1,X_S2],axis=0)

    y_S1 = np.repeat(0, int(X_S1_trials.shape[0]))
    y_S2 = np.repeat(1, int(X_S2_trials.shape[0]))

    y_trials = np.concatenate((y_S1, y_S2))

    return X_trials, y_trials
