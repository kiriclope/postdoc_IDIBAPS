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

shade_alpha      = 0.2
lines_alpha      = 0.8
pal = sns.color_palette('husl', 9)
pal = ['r','b','y']

trial_types = ['ND_S1','ND_S2','D1_S1','D1_S2','D2_S1','D2_S2']

trial_types = gv.trials

for gv.mouse in [gv.mice[2]] : 

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
        time = np.linspace(0, gv.duration, X.shape[2]) ; 

        F0 = np.mean(X[:,:,gv.bins_baseline],axis=2) 
        F0 = F0[:,:, np.newaxis]
        
        X = (X-F0) / (F0 + 0.0000001) 

        n_trials = X.shape[0]
        n_neurons = X.shape[1] 
        trial_size = X.shape[2]  
        
        trial_averages = []
        for gv.trial in gv.trials:
            X_S1, X_S2 = data.get_S1_S2_trials(X, y) 

            print('X_S1', X_S1.shape, 'X_S2', X_S2.shape)

            X_S1_S2 = np.vstack((X_S1, X_S2))
            X_S1_S2 = np.mean(X_S1_S2, axis=0)
            
            # X_S1 = np.mean(X_S1, axis=0)
            # X_S2 = np.mean(X_S2, axis=0)
            # X_S1_S2 = np.hstack((X_S1, X_S2))
            
            print('X_S1_S2', X_S1_S2.shape)

            trial_averages.append(X_S1_S2)
        
        Xa = np.hstack(trial_averages)        
        Xa = prep.z_score(Xa) #Xav_sc = center(Xav)
        print('Xa', Xa.shape)
    
        pca = PCA(n_components=3)
        Xa_p = pca.fit_transform(Xa.T).T
        print('explained_variance', pca.explained_variance_ratio_)
        
        fig, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey='row')
        for comp in range(3):
            ax = axes[comp]
            for kk, type in enumerate(trial_types):
                # print('pca',comp,'trial',type)
                x = Xa_p[comp, kk * trial_size :(kk+1) * trial_size]
                x = gaussian_filter1d(x, sigma=3)
                ax.plot(time, x, c=pal[kk], label=type)            
            ax.set_xlim([0, gv.t_test[1]+1])
            add_stim_to_plot(ax)
            ax.set_ylabel('PC {}'.format(comp+1))
            add_orientation_legend(axes[2])
        axes[1].set_xlabel('Time (s)')
        sns.despine(fig=fig, right=True, top=True)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        ax.legend()
        
    # find the indices of the three largest elements of the second eigenvector
    # units = np.abs(pca.components_[1, :].argsort())[::-1][0:3]
    # print(units)
    # f, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey=False,
    #                        sharex=True)
    # for ax, unit in zip(axes, units):
    #     ax.set_title('Neuron {}'.format(unit))
    #     for t, ind in enumerate(t_type_ind):
    #         x = np.array(trials)[ind][:, unit, :]
    #         sns.tsplot(x, time=time,
    #                    ax=ax,
    #                    err_style='ci_band',
    #                    ci=95,
    #                    color=pal[t])
        
    # axes[1].set_xlabel('Time (s)')
    # sns.despine(fig=f, right=True, top=True)
