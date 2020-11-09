from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 

import data.constants as gv 
import data.utils as data 

import plotting as plot 

import decode.cross_temp.utils as decode 
importlib.reload(decode) ; 

def temporal_decoder(X_proj, NO_PCA=0, IF_EPOCHS=0):
    
    if IF_EPOCHS:
        gv.epochs = ['ED','MD','LD'] 
    else:
        gv.epochs = ['all'] 
        
    C=1 
    penalty='l2' 
    # clf = LogisticRegression(C=C, solver='liblinear', penalty=penalty,tol=1e-6, max_iter=int(1e6)) 
    clf = svm.LinearSVC(C=C, penalty=penalty, loss='squared_hinge', dual=False, tol=1e-6, max_iter=int(1e6), fit_intercept=False) 

    for i, gv.trial in enumerate(gv.trials): 
        X_S1_trials = X_proj[i,0] 
        X_S2_trials = X_proj[i,1] 
        X_trials, y_trials = data.get_X_y_epochs(X_S1_trials, X_S2_trials)  
        print('trial:', gv.trial, 'X', X_trials.shape,'y', y_trials.shape) 
        
        # scores = decode.cross_temp_clf_par(clf, X_trials, y_trials, 1, cv=10) 
        scores, scores_std = decode.mne_cross_temp_clf(X_trials, y_trials, clf, cv=20) 
        print('scores', scores.shape) 
        decode.cross_temp_plot_mat(scores, IF_EPOCHS) 
        
        figname = '%s_session_%s_trial_%s_cross_temp_decoder' % (gv.mouse,gv.session,gv.trial) 

        figdir = plot.figDir()
        
        if NO_PCA: 
            figdir = figdir + '/no_pca/'
        if IF_EPOCHS:            
            figdir = figdir + '/epochs/'
        
        if not os.path.isdir(figdir):
            os.makedirs(figdir)
            
        plot.save_fig(figname, figdir) 
