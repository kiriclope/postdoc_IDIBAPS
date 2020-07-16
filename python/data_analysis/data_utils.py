import numpy as np

def which_trials(y_labels, trials):
    y_trials = []
    
    if 'ND' in trials:
        y_trials = np.argwhere( (y_labels[4]==0) & (y_labels[8]==0) & ( (y_labels[2]==1) | (y_labels[2]==4) ) ).flatten()
        # y_trials = np.argwhere( (y_labels[4]==0) & (y_labels[8]==0) ).flatten()
        if 'S1' in trials:
            y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[4]==0) & (y_labels[8]==0) ).flatten()
        elif 'S2' in trials:
            y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[4]==0) & (y_labels[8]==0) ).flatten()

    elif 'D1' in trials:
        y_trials = np.argwhere((y_labels[4]==13) & (y_labels[8]==0) ).flatten() 
        if 'S1' in trials:
            y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[4]==13) & (y_labels[8]==0) ).flatten()
        elif 'S2' in trials:
            y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[4]==13) & (y_labels[8]==0) ).flatten()
            
    elif 'D2' in trials:
        y_trials = np.argwhere((y_labels[4]==14) & (y_labels[8]==0)).flatten()
        if 'S1' in trials:
            y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[4]==14) & (y_labels[8]==0) ).flatten()
        elif 'S2' in trials:
            y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[4]==14) & (y_labels[8]==0) ).flatten()

    elif 'paired' in trials :
        y_trials = np.argwhere( (y_labels[4]==0) & (y_labels[8]==0) &
                                          ((y_labels[0]==17) & (y_labels[1]==11)) |
                                          ((y_labels[0]==18) & (y_labels[1]==12)) ).flatten()
    elif 'unpaired' in trials :
        y_trials = np.argwhere( (y_labels[4]==0) & (y_labels[8]==0) &
                                          ((y_labels[0]==17) & (y_labels[1]==12)) | 
                                          ((y_labels[0]==18) & (y_labels[1]==11)) ).flatten()         
        
    return y_trials

################################################################################
################################################################################

def bin_data(data, bin_step, bin_size):
    # bin_step number of pts btw bins, bin_size number of size in each bin
    bin_array = [np.mean(np.take(data,np.arange(int(i*bin_step),int(i*bin_step+bin_size)), axis=2), axis=2) for i in np.arange(data.shape[2]//bin_step-1)]
    bin_array = np.array(bin_array)
    bin_array = np.rollaxis(bin_array,0,3)
    return bin_array
