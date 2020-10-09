import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import mne
from mne.datasets import sample
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)

def cross_temp_val(clf,X,y,scoring='accuracy'):

    time_gen = GeneralizingEstimator(clf, n_jobs=-1, scoring=scoring, verbose=True)
    scores = cross_val_multiscore(time_gen, X, y, cv=5, n_jobs=-1)
    mean_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0)

    return mean_scores, std_scores

def plot_cross_temp_mat(scores, interpolation='lanczos', cmap='jet'): 
    ig, ax = plt.subplots(1, 1) 
    im = ax.imshow(scores, interpolation='lanczos',cmap='jet', origin='lower', vmin=.5, vmax=1, extent = [-2 , 18, -2 , 18])
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Temporal generalization')
    plt.colorbar(im, ax=ax)
    
    plt.axvline(x=0, c='k', ls='-') 
    plt.axhline(y=0, c='k', ls='-')  

    plt.axvline(x=1, c='k', ls='--') # DPA early delay 
    plt.axvline(x=2.5, c='k', ls='--') 

    plt.axvline(x=3.5, c='r', ls='--') # DRT delay
    plt.axvline(x=4.5, c='r', ls='--') 

    plt.axvline(x=5.5, c='k', ls='--') # DPA late delay
    plt.axvline(x=7, c='k', ls='--') 

    plt.axhline(y=1, c='k', ls='--') # DPA early delay 
    plt.axhline(y=2.5, c='k', ls='--') 

    plt.axhline(y=3.5, c='r', ls='--') # DRT delay
    plt.axhline(y=4.5, c='r', ls='--') 

    plt.axhline(y=5.5, c='k', ls='--') # DPA late delay
    plt.axhline(y=7, c='k', ls='--') 
    
    plt.xlim([-2, 10]);
    plt.ylim([-2, 10]);
