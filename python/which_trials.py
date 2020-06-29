import numpy as np

def which_trials(y_labels, trials):
    y_trials = [] 
    if 'ND_trials' in trials:
        y_trials = np.argwhere((y_labels[4]==0) & (y_labels[8]==0)).flatten()        
        if 'S1' in trials:
            y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[4]==0) & (y_labels[8]==0) ).flatten()
        elif 'S2' in trials:
            y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[4]==0) & (y_labels[8]==0) ).flatten()

    elif 'D1_trials' in trials:
        y_trials = np.argwhere((y_labels[4]==13) & (y_labels[8]==0)).flatten()
        if 'S1' in trials:
            y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[4]==13) & (y_labels[8]==0) ).flatten()
        elif 'S2' in trials:
            y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[4]==13) & (y_labels[8]==0) ).flatten()
            
    elif 'D2_trials' in trials:
        y_trials = np.argwhere((y_labels[4]==14) & (y_labels[8]==0)).flatten()
        if 'S1' in trials:
            y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[4]==14) & (y_labels[8]==0) ).flatten()
        elif 'S2' in trials:
            y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[4]==14) & (y_labels[8]==0) ).flatten()
            
    return y_trials

# y_paired_ND_trials = np.argwhere( ((y_frame_labels[0][y_ND_trials]==17) & (y_frame_labels[1][y_ND_trials]==11)) | 
#                                  ((y_frame_labels[0][y_ND_trials]==18) & (y_frame_labels[1][y_ND_trials]==12)) ).flatten() 
# print(y_paired_ND_trials.shape)
# print(y_paired_ND_trials)

# y_non_paired_ND_trials = np.argwhere( ((y_frame_labels[0][y_ND_trials]==17) & (y_frame_labels[1][y_ND_trials]==12)) | 
#                                  ((y_frame_labels[0][y_ND_trials]==18) & (y_frame_labels[1][y_ND_trials]==11)) ).flatten() 
# print(y_non_paired_ND_trials.shape)
# print(y_non_paired_ND_trials)
