from libs import * 

sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 

import data.constants as gv 
importlib.reload(gv) ; 

import data.utils as data 
importlib.reload(data) ; 

import data.fct_facilities as fac
importlib.reload(fac) ; 
fac.SetPlotParams() 

import preprocessing as prep
from ploting import *

pal = ['r','b','y']

trial_types = gv.trials
n_components=3
gv.laser_on = 1

for gv.mouse in [gv.mice[1]] : 

    data.get_sessions_mouse() 
    data.get_stimuli_times() 
    data.get_delays_times()
    
    for gv.session in [gv.sessions[-1]] : 
        X, y = data.get_fluo_data()
        print('mouse', gv.mouse, 'session', gv.session, 'data X', X.shape,'y', y.shape) 
        
        data.get_delays_times() 
        data.get_frame_rate() 
        data.get_bins(t_start=0) 

        F0 = np.mean(X[:,:,gv.bins_baseline],axis=2)
        F0 = F0[:,:, np.newaxis]

        idx = np.where(F0==0)[1]
        F0 = np.delete(F0, idx, axis=1) 
        X = np.delete(X, idx, axis=1) 

        X = (X -F0) / (F0 + 0.0000001) 
        
        gv.duration = X.shape[2]/gv.frame_rate 
        time = np.linspace(0, gv.duration, X.shape[2]) ; 

        n_trials = X.shape[0] 
        n_neurons = X.shape[1] 
        trial_size = X.shape[2] 

        trials = [] 
        trials_avg = [] 
        for gv.trial in gv.trials: 
            X_S1, X_S2 = data.get_S1_S2_trials(X, y) 

            print('X_S1', X_S1.shape, 'X_S2', X_S2.shape)

            X_S1_S2 = np.vstack((X_S1, X_S2))
            X_S1_S2_avg = np.mean(X_S1_S2, axis=0)

            print('X_S1_S2', X_S1_S2.shape)
            print('X_S1_S2_avg', X_S1_S2_avg.shape)

            trials.append(X_S1_S2)
            trials_avg.append(X_S1_S2_avg)

        n_trials = X_S1_S2.shape[0]
        trial_type = ['ND'] * n_trials + ['D1'] * n_trials + ['D2'] * n_trials
        
        X_trials = np.vstack(trials)
        print('X_trials', X_trials.shape)
        
        X_avg = np.hstack(trials_avg) 
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_avg = scaler.fit_transform(X_avg.T).T
        print('X_avg', X_avg.shape)
        
        pca = PCA(n_components=n_components)
        pca.fit_transform(X_avg.T)
        print('explained_variance', pca.explained_variance_ratio_)

        projected_trials = []
        for i in range(X_trials.shape[0]):
            # scale every trial using the same scaling applied to the averages 
            trial = scaler.transform(X_trials[i,:,:].T).T 
            # project every trial using the pca fit on averages
            proj_trial = pca.transform(trial.T).T
            projected_trials.append(proj_trial)
            
        gt = {comp : {t_type : [] for t_type in gv.trials} for comp in range(n_components)}

        for comp in range(n_components): 
            for i, t_type in enumerate(trial_type): 
                t = projected_trials[i][comp, :] 
                gt[comp][t_type].append(t) 
                
        for comp in range(n_components):
            for t_type in trial_types: 
                gt[comp][t_type] = np.vstack(gt[comp][t_type]).transpose() 
            
        f, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey=True, sharex=True)
        x = time
        for comp in range(3):
            ax = axes[comp]
            for t, t_type in enumerate(trial_types):
                y = np.mean(gt[comp][t_type],axis=1) 
                ax.plot(x, y, color=pal[t])
                ci = prep.conf_inter(gt[comp][t_type]) 
                ax.fill_between(x, ci[0], ci[1] , color=pal[t], alpha=.1) 
                add_stim_to_plot(ax) 
                ax.set_xlim([0, gv.t_test[1]+1])
            ax.set_ylabel('PC {}'.format(comp+1))
            axes[1].set_xlabel('Time (s)')
            sns.despine(right=True, top=True)
            add_orientation_legend(axes[2])
