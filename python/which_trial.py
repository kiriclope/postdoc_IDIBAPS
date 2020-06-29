def which_trials(y_labels, trials):
    if trials == 'ND_trials':
        y_trials = np.argwhere((y_labels[4]==0) & (y_labels[8]==0)).flatten()
    elif trials == 'D1_trials':
        y_trials = np.argwhere((y_labels[4]==13) & (y_labels[8]==0)).flatten()
    elif trials == 'D2_trials':
        y_trials = np.argwhere((y_labels[4]==14) & (y_labels[8]==0)).flatten()
    return y_trials

