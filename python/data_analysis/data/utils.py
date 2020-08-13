from .std_lib import *
from . import global_vars as gv

def get_delays_times():
    if((gv.folder=='ChRM04') | (gv.folder=='JawsM15')):
        gv.t_early_delay = [3, 4.5]
        gv.t_DRT_delay = [5.5, 6.5]
        gv.t_late_delay = [7.5, 9]
    else:
        gv.t_early_delay = [3, 6]
        gv.t_DRT_delay = [7, 8]
        gv.t_late_delay = [9, 12]


def get_stimuli_times():
    if((gv.folder=='ChRM04') | (gv.folder=='JawsM15')):
        gv.t_distractor = [4.5, 5.5]
        gv.t_cue = [6.5, 7]
        gv.t_DRT_reward = [7, 7.5]
        gv.t_test = [9, 10]
    else:
        gv.t_distractor = [6, 7]
        gv.t_cue = [8, 8.5]
        gv.t_DRT_reward = [8.5, 9]
        gv.t_test = [12, 13]

def get_frame_rate():
    if((gv.folder=='ChRM04') | (gv.folder=='JawsM15')):
        gv.frame_rate = 6
    else :
        gv.frame_rate = 7.5

def get_sessions_folder():
    if gv.folder=='C57_2_DualTask' :
        gv.sessions = list( map( str, np.arange(20200116, 20200121) ) )
    elif gv.folder=='ChRM04' :
        gv.sessions = list( map( str, np.arange(20200521, 20200527) ) )
    elif gv.folder=='JawsM15' :
        gv.sessions = list( map( str, np.arange(20200605, 20200610) ) )


def get_fluo_data():

    if((gv.folder=='ChRM04') | (gv.folder=='JawsM15')):
        data = loadmat('/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/DataForAlexandre/' + gv.folder + '/' + gv.session + 'SumFluoTraceFile' + '.mat')
        
        # X_fluo = np.rollaxis(data['C_df'],1,0)
        X_data = np.rollaxis(data['dFF0'],1,0)
        # X_rates = np.rollaxis(data['S_dec'],1,0)
        y_labels = data['Events'].transpose()
        gv.frame_rate = 6
    
    else:
        data = loadmat('/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/DataForAlexandre/' + gv.folder +  '/' + gv.session + '-C57-2-DualTaskAcrossDaySameROITrace' + '.mat')
        data_labels = loadmat('/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/DataForAlexandre/' + gv.folder + '/' + gv.session + '-C57-2-DualTask-SumFluoTraceFile' + '.mat')
        X_data = np.rollaxis(data['SameAllCdf'],2,0)
        y_labels= data_labels['AllFileEvents'+gv.session][0][0][0].transpose()
        gv.frame_rate = 7.5

    gv.duration = X_data.shape[2]/gv.frame_rate
    gv.time = np.linspace(0,gv.duration,X_data.shape[2]);
    gv.bins = np.arange(0,len(gv.time))
    
    return X_data, y_labels

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

    trial = gv.trial
    gv.trial = trial + "_S1"
    y_S1_trials = which_trials(y_labels)

    gv.trial = trial + "_S2"
    y_S2_trials = which_trials(y_labels)

    gv.trial = trial
    X_S1_trials = X_data[y_S1_trials]
    X_S2_trials = X_data[y_S2_trials]

    return X_S1_trials, X_S2_trials

def get_bins_ED_LD():
    # gv.bins_ED = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_early_delay[1]-.5) & (gv.time[bin]<=gv.t_early_delay[1]) ]
    # gv.bins_LD = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_late_delay[1]-.5) & (gv.time[bin]<=gv.t_late_delay[1]) ]
    # gv.bins_ED = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_early_delay[0]) & (gv.time[bin]<=gv.t_early_delay[1]) ]
    # gv.bins_LD = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_late_delay[0]) & (gv.time[bin]<=gv.t_late_delay[1]) ]

    gv.bins_ED = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_early_delay[0] + (gv.t_early_delay[1]-gv.t_early_delay[0])/2) & (gv.time[bin]<=gv.t_early_delay[1]) ]
    gv.bins_LD = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_late_delay[0] + (gv.t_late_delay[1]-gv.t_late_delay[0])/2) & (gv.time[bin]<=gv.t_late_delay[1]) ]

    # gv.bins_ED = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_sample[0]) & (gv.time[bin]<=gv.t_sample[1]) ]
    # gv.bins_LD = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_early_delay[0] + (gv.t_early_delay[1]-gv.t_early_delay[0])/2) & (gv.time[bin]<=gv.t_early_delay[1]) ]

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
    X_S1_trials, X_S2_trials = get_S1_S2_trials(X_data, y_labels)
    # print(y_S1_trials[0])
    X_S1 = bin_data(X_S1_trials, gv.n_bin, gv.n_bin)
    X_S2 = bin_data(X_S2_trials, gv.n_bin, gv.n_bin)
    # print(X_S1[0])
    X_trials = np.concatenate([X_S1,X_S2],axis=0)
    # print(X_trials[0])
    y_S1 = np.repeat(0, int(X_S1_trials.shape[0]))
    y_S2 = np.repeat(1, int(X_S2_trials.shape[0]))

    y_trials = np.concatenate((y_S1, y_S2))

    return X_trials, y_trials

def plot_avg(X):
    time = np.linspace(0, gv.duration, X.shape[2]);
    X_window = np.mean(X, axis=1) # avg over neurons 
    X_avg = np.mean(X_window,axis=0) # avg over trials 
    X_std = np.std(X_window,axis=0) # std over trials

    plt.plot(time,X_avg,c='k')
    plt.fill_between(time, X_avg - X_std, X_avg + X_std, alpha=0.25, color='magenta')
    
    plt.xlabel('t (s)') ;
    plt.ylabel('$\Delta F$')

def vplot_delays():
    plt.gcf() 
    plt.axvline(x=gv.t_sample[0], c='k', ls='--') # DPA sample onset 
    plt.axvline(x=gv.t_sample[1], c='k', ls='--')

    plt.axvline(x=gv.t_test[0], c='k', ls='--') # DPA test onset
    plt.axvline(x=gv.t_test[1], c='k', ls='--')
    
    plt.axvline(x=gv.t_distractor[0], c='r', ls='--') # DRT distractor onset
    plt.axvline(x=gv.t_distractor[1], c='r', ls='--')

    plt.axvline(x=gv.t_cue[0], c='r', ls='--') # DRT cue onset
    plt.axvline(x=gv.t_cue[1], c='r', ls='--')
