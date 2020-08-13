import sys
sys.path.append('../../')

from .std_lib import *
from .sklearn_lib import *
from .mne_lib import *

import data.global_vars as gv

def cross_temp_clf(X, y, clf=None, scoring='accuracy', cv=5):

    if(clf==None):
        clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs'))
        # clf = make_pipeline(StandardScaler(), svm.SVC(kernel='linear'))
    
    time_gen = GeneralizingEstimator(clf, n_jobs=-1, scoring=scoring, verbose=False)
    scores = cross_val_multiscore(time_gen, X, y, cv=cv, n_jobs=-1)
    scores = np.mean(scores, axis=0)
    # scores_std= np.std(scores, axis=0)

    return scores

def cross_temp_plot_diag(scores):
    time = np.linspace(0, gv.duration, scores.shape[0]);
    diag_scores = np.diag(scores)
    diag_scores_std = scores_std

    plt.plot(time, diag_scores)
    plt.fill_between(time, diag_scores - diag_scores_std, diag_scores + diag_scores_std, alpha=0.25, color='green')

    y_for_chance = np.repeat(0.50, len(diag_scores) ) ;
    plt.plot(time, y_for_chance, '--', c='black')
    plt.ylim([0, 1])

    plt.axvline(x=2, c='black', linestyle='dashed')
    plt.axvline(x=3, c='black', linestyle='dashed')

    plt.axvline(x=4.5, c='r', linestyle='dashed')
    plt.axvline(x=5.5, c='r', linestyle='dashed')

    plt.axvline(x=6.5, c='r', linestyle='dashed')
    plt.axvline(x=7, c='r', linestyle='dashed')
    
    plt.text(2., 1., 'Sample', rotation=0)
    plt.text(9., 1., 'Test', rotation=0)

    plt.axvline(x=9, c='black', linestyle='dashed')
    plt.axvline(x=10, c='black', linestyle='dashed')
    
    plt.xlim([0,gv.duration]) ;

def cross_temp_plot_mat(scores):

    duration = scores.shape[0]/gv.frame_rate 

    ig, ax = plt.subplots(1, 1)
    im = ax.imshow(scores, interpolation='lanczos',cmap='jet', origin='lower', vmin=0.5, vmax=1, extent = [-2 , gv.duration-2, -2 , gv.duration-2])
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title(gv.trial)
    plt.colorbar(im, ax=ax)

    plt.axvline(x=gv.t_early_delay[0]-2, c='k', ls='--')
    plt.axvline(x=gv.t_early_delay[1]-2, c='k', ls='--') # DPA early delay
    
    plt.axvline(x=gv.t_DRT_delay[0]-2, c='r', ls='--') #DRT delay
    plt.axvline(x=gv.t_DRT_delay[1]-2, c='r', ls='--') 

    plt.axvline(x=gv.t_late_delay[0]-2, c='k', ls='--')
    plt.axvline(x=gv.t_late_delay[1]-2, c='k', ls='--') # DPA late delay
    
    plt.axhline(y=gv.t_early_delay[0]-2, c='k', ls='--')
    plt.axhline(y=gv.t_early_delay[1]-2, c='k', ls='--') # DPA early delay

    plt.axhline(y=gv.t_DRT_delay[0]-2, c='r', ls='--') #DRT delay
    plt.axhline(y=gv.t_DRT_delay[1]-2, c='r', ls='--') 
    
    plt.axhline(y=gv.t_late_delay[0]-2, c='k', ls='--')
    plt.axhline(y=gv.t_late_delay[1]-2, c='k', ls='--') # DPA late delay

    plt.xlim([gv.t_early_delay[0]-2, gv.t_late_delay[1]]);
    plt.ylim([gv.t_early_delay[0]-2, gv.t_late_delay[1]]);

    plt.xlim([-2, gv.duration-2]);
    plt.ylim([-2, gv.duration-2]);

    fig = plt.gcf()
    fig.set_size_inches(1.33*5, 5)
    fig.savefig('../../../figs/' + gv.folder + '/' + gv.session + '_' + gv.trial + '_cross_temp_' + str(round(gv.n_bin/gv.frame_rate, 3)) + 's_bin_' + '.svg', format='svg')
