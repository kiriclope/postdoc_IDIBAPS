import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D

import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#####################################################################
#####################################################################

def pca_func(X_Window,n_pca):
    # reshaping the array for samples = trials features = neurons by time features
    # X_Window  = np.reshape((X_Window), (X_Window.shape[0], X_Window.shape[1]* X_Window.shape[2]))
    #Standardizing the features
    X_Window = StandardScaler().fit_transform(X_Window)  
    # PCA projection to 2D
    pca = PCA(n_components=n_pca)
    principal_components = pca.fit_transform(X_Window)
    principal_components = principal_components.transpose()
    explained_variance = pca.explained_variance_ratio_
    return principal_components, explained_variance

#####################################################################
#####################################################################

def pca_trajectory(X_trials,time_window,n_pca):
    split_windows = np.array_split(X_trials, X_trials.shape[0]/time_window, axis=1)
    i=0
    for X_window in split_windows:
        principal_components, explained_variance = pca_func(X_window,n_pca)
        mean_pc = np.mean(principal_components,axis=1)
        if i==0:
          pc_vec = mean_pc[:,np.newaxis]
          i=i+1
        else:
          pc_vec = np.concatenate([pc_vec, mean_pc[:,np.newaxis]],axis=1)
    return pc_vec

#####################################################################
#####################################################################

def pca_2D(pca_delay_1_trials, pca_delay_2_trials):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    ax.scatter(pca_delay_1_trials[0],pca_delay_1_trials[1],c='b')
    ax.scatter(pca_delay_2_trials[0],pca_delay_2_trials[1],c='r')

#####################################################################
#####################################################################
def pca_3D(pca_delay_1_trials, pca_delay_2_trials):
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, 1, 1])
    ax.scatter(pca_delay_1_trials[0], pca_delay_1_trials[1], pca_delay_1_trials[2], c='b')
    ax.scatter(pca_delay_2_trials[0], pca_delay_2_trials[1], pca_delay_1_trials[2], c='r')

#####################################################################
#####################################################################

def pca_plot(X_Delay_1_trials, X_Delay_2_trials, n_pca):
    pca_delay_1_trials = pca_func(X_Delay_1_trials,n_pca)
    pca_delay_2_trials = pca_func(X_Delay_2_trials,n_pca)

    print(pca_delay_1_trials[1],pca_delay_2_trials[1])
    if n_pca==2:
      pca_2D(pca_delay_1_trials[0],pca_delay_2_trials[0])
    else:
      pca_3D(pca_delay_1_trials[0],pca_delay_2_trials[0])
