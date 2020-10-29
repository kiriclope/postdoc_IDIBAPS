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
from plotting import * 

pal = ['r','b','y'] 
gv.samples = ['S1', 'S2'] 
# gv.samples = ['all'] 

gv.trials = ['ND','D1','D2']
gv.laser_on = 0
n_components = 3
AVG_EPOCHS = 0 

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
        
        gv.duration = X.shape[2]/gv.frame_rate 
        gv.time = np.linspace(0, gv.duration, X.shape[2]) 
        
        F0 = np.mean(X[:,:,gv.bins_baseline], axis=2)        
        F0 = F0[:,:, np.newaxis]

        idx = np.where(F0<=0.01)[1] 
        F0 = np.delete(F0, idx, axis=1) 
        X = np.delete(X, idx, axis=1) 

        X = (X-F0) / (F0 + 0.0000001) 
        
        trial_averages = []
        for gv.trial in gv.trials:
            X_S1, X_S2 = data.get_S1_S2_trials(X, y) 
            data.get_trial_types(X_S1) 

            print('X_S1', X_S1.shape, 'X_S2', X_S2.shape) 

            if AVG_EPOCHS : 
                X_S1 = data.avgOverEpochs(X_S1) 
                X_S2 = data.avgOverEpochs(X_S2) 
                gv.trial_size = len(gv.epochs) 
                
            if 'all' in gv.samples: 
                # concatenate S1 and S2 trials
                X_S1_S2 = np.vstack((X_S1, X_S2)) 
                X_S1_S2 = np.mean(X_S1_S2, axis=0) 
            else: 
                # keep S1 and S2 trials separated                
                X_S1 = np.mean(X_S1, axis=0) 
                X_S2 = np.mean(X_S2, axis=0) 
                X_S1_S2 = np.hstack((X_S1, X_S2)) 
            
            print('X_S1_S2', X_S1_S2.shape) 

            trial_averages.append(X_S1_S2) 
        
        X_avg = np.hstack(trial_averages) 
        X_avg = prep.z_score(X_avg) # Xav_sc = center(Xav)
        print('X_avg', X_avg.shape) 
    
        pca = PCA(n_components=n_components)
        X_avg_pc = pca.fit_transform(X_avg.T).T
        print('explained_variance', pca.explained_variance_ratio_)

        X_avg_pc = np.asarray(X_avg_pc)
        print(X_avg_pc.shape)
        X_avg_pc = X_avg_pc.reshape(n_components, len(gv.trials), len(gv.samples), gv.trial_size) 
        print(X_avg_pc.shape)
        
        fig, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey='row')
        for pc in range(3):
            ax = axes[pc] 
            for i, trial in enumerate(gv.trials):
                for j, sample in enumerate(gv.samples):
                    y = X_avg_pc[pc, i, j] 
                    y = gaussian_filter1d(y, sigma=3) 

                    ax.plot(gv.time, y, c=pal[i], label= trial+'_'+sample) 
                    ax.set_xlim([0, gv.t_test[1]+1])
                    
                    add_stim_to_plot(ax)
                    ax.set_ylabel('PC {}'.format(pc+1))
        add_orientation_legend(axes[2])
        axes[1].set_xlabel('Time (s)')
        sns.despine(fig=fig, right=True, top=True)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        add_orientation_legend(axes[2])
