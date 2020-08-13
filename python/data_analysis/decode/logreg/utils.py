import sys
sys.path.append('../../')

from .std_lib import *
from .sklearn_lib import *

import data.global_vars as gv
import data.utils as func

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def log_reg(X_trials, y_trials, shuffle=0, clf=None):
    """ Returns coefficient of the logistic regression """    
    if(clf==None):
        clf = LogisticRegression()
        #clf = LogisticRegression(C=1e15,solver='lbfgs')
    coefs = []
    for bin in np.arange(0,X_trials.shape[0]):
        X = X_trials[bin]
        X = StandardScaler().fit_transform(X)
        # y = y_trials
        
        if bin==0:
            y = y_trials
            if shuffle:
                random.shuffle(y)
        
        clf.fit(X, y)
        print('score', clf.score(X,y))
        coefs.append( clf.coef_.flatten() ) 
    
    coefs = np.asarray(coefs)
    return coefs

def get_cos_trials(X_data, y_labels, shuffle=0):
    """ Returns the cosine of the angle alpha between 
    the normal vector defining the early delay log-reg 
    and the one defining the late delay log-reg """
    cos_alp_trials = []
    alpha_trials = []
    for gv.trial in gv.trials:
        X_S1_trials, X_S2_trials = func.get_S1_S2_trials(X_data, y_labels)        
        X_trials, y_trials = func.get_X_y_ED_LD(X_S1_trials, X_S2_trials)

        # if shuffle:
        #     random.shuffle(y_trials)
            
        coefs = log_reg(X_trials, y_trials, shuffle=shuffle)
        # print(coefs)
        alpha, cos_alp = get_cos(coefs)

        alpha_trials.append(alpha[0])
        cos_alp_trials.append(cos_alp[0])
        
    return alpha_trials, cos_alp_trials

def get_cos(coefs) :
    """ Returns the cosine of the angle alpha between vector coefs[0] (early delay) and coefs[1] (late delay) """
    alphas = []
    cos_alp=[]
    for i in np.arange(0,coefs.shape[0]):
        for j in np.arange(i+1,coefs.shape[0]):
            alpha = angle_between(coefs[i], coefs[j])
            alphas.append(alpha)
            cos_alp.append(np.cos(alpha))
    # print(cos_alp)
    return alphas, cos_alp

def plot_cos_bar(mean_cos, std_cos):
    """ bar plot of the value of the cosine of alpha """
    labels = gv.trials ;
    xticks = np.arange(len(labels))
    width = .2
    
    fig, ax = plt.subplots()
    rects = ax.bar(xticks, mean_cos, width, label=gv.folder, yerr=std_cos) ;
    ax.set_xticks(xticks) ;
    ax.set_xticklabels(labels) ;
    ax.legend()

    plt.xlabel('trials')
    plt.ylabel('cos($\\alpha$) late vs early')

def get_z_score_cos_alp(cos_alp, mean_cos, std_cos):
    """ bar plot of the z-score : (observation - mean shuffle)/std_shuffle of the cosine of alpha """
    z_score_cos_alp = [ (cos_alp[i] - mean_cos[i]) / std_cos[i] for i in range(0,3) ]
    print(z_score_cos_alp)

    labels = gv.trials ;
    xticks = np.arange(len(labels))
    width = .2
    
    fig, ax = plt.subplots()

    ax.bar(xticks, z_score_cos_alp, width, label=gv.folder, color='b') ;
    ax.set_xticks(xticks) ;
    ax.set_xticklabels(labels) ;
    ax.legend(loc='best')

    plt.xlabel('trials')
    plt.ylabel('cos($\\alpha$)')
    plt.ylabel('z-score of cos($\\alpha$)')

    return z_score_cos_alp

def get_z_score_alpha(alpha, mean_alpha, std_alpha):
    """ bar plot of the z-score : (observation - mean shuffle)/std_shuffle of alpha """
    z_score_alpha = [ (alpha[i] - mean_alpha[i]) / std_alpha[i] for i in range(0,3) ]
    print(z_score_alpha)

    labels = gv.trials ;
    xticks = np.arange(len(labels))
    width = .2
    
    fig, ax = plt.subplots()

    ax.bar(xticks, z_score_alpha, width, label=gv.folder, color='b') ;
    ax.set_xticks(xticks) ;
    ax.set_xticklabels(labels) ;
    ax.legend(loc='best')

    plt.xlabel('trials')
    plt.ylabel('$\\alpha$')
    plt.ylabel('z-score of $\\alpha$')

    return z_score_alpha

def get_shuffle_cos_trials(n_shuffle=1000):
    """ Returns the mean, std and matrix of the cosine of alpha when the labels are shuffled n_shuffle times"""
    mat_cos = Parallel(n_jobs=56)(delayed(get_cos_trials)(X_data, y_labels, shuffle=1) for i in range(n_shuffle))
    mat_cos = np.asarray(mat_cos)
    print(mat_cos.shape)

    mean_cos = np.mean(mat_cos, axis=0)
    std_cos = np.std(mat_cos, axis=0)
    
    print(mean_cos)
    print(std_cos)

    return mean_cos, std_cos, mat_cos
