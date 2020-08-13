import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn import svm
from sklearn.preprocessing import StandardScaler
import sklearn.ensemble
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.model_selection import cross_validate

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

import mne
from mne.datasets import sample
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)

import data
from data.files import *

def cross_temp_clf(X, y, clf=None, scoring='accuracy', cv=5):

    if(clf==None):
        # clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs'))
        clf = make_pipeline(StandardScaler(), svm.SVC(kernel='linear'))
    
    time_gen = GeneralizingEstimator(clf, n_jobs=-1, scoring=scoring, verbose=False)
    scores = cross_val_multiscore(time_gen, X, y, cv=cv, n_jobs=-1)
    scores = np.mean(scores, axis=0)
    scores_std= np.std(scores, axis=0)

    return scores, scores_std

def cross_temp_plot_diag(scores):
    time = np.linspace(0,duration,scores.shape[0]);
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
    
    plt.xlim([0,duration]) ;

def cross_temp_plot_mat(scores):
    global folder, session, trial 
    global frame_rate, duration, n_bin 
    global t_early_delay, t_DRT_delay, t_late_delay 

    duration = scores.shape[0]/frame_rate 

    ig, ax = plt.subplots(1, 1)
    im = ax.imshow(scores, interpolation='lanczos',cmap='jet', origin='lower', vmin=0.5, vmax=1, extent = [-2 , duration-2, -2 , duration-2])
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title(trial)
    plt.colorbar(im, ax=ax)

    plt.axvline(x=t_early_delay[0]-2, c='k', ls='--')
    plt.axvline(x=t_early_delay[1]-2, c='k', ls='--') # DPA early delay
    
    plt.axvline(x=t_DRT_delay[0]-2, c='r', ls='--') #DRT delay
    plt.axvline(x=t_DRT_delay[1]-2, c='r', ls='--') 

    plt.axvline(x=t_late_delay[0]-2, c='k', ls='--')
    plt.axvline(x=t_late_delay[1]-2, c='k', ls='--') # DPA late delay
    
    plt.axhline(y=t_early_delay[0]-2, c='k', ls='--')
    plt.axhline(y=t_early_delay[1]-2, c='k', ls='--') # DPA early delay

    plt.axhline(y=t_DRT_delay[0]-2, c='r', ls='--') #DRT delay
    plt.axhline(y=t_DRT_delay[1]-2, c='r', ls='--') 
    
    plt.axhline(y=t_late_delay[0]-2, c='k', ls='--')
    plt.axhline(y=t_late_delay[1]-2, c='k', ls='--') # DPA late delay

    plt.xlim([t_early_delay[0]-2, t_late_delay[1]]);
    plt.ylim([t_early_delay[0]-2, t_late_delay[1]]);

    plt.xlim([-2, duration-2]);
    plt.ylim([-2, duration-2]);

    fig = plt.gcf()
    fig.set_size_inches(1.33*5, 5)
    fig.savefig('../figs/' + folder + '/' + session + '_' + trial + '_cross_temp_' + str(round(bin/frame_rate, 3)) + 's_bin_' + '.svg',
                format='svg')

