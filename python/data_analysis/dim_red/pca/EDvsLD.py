from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 
from scipy.spatial import distance

import data.constants as gv 
import plotting as plot 

from joblib import Parallel, delayed
import multiprocessing

def bootstrap_par(S1,S2):
    if gv.n_boot==1: 
        idx = np.arange(0, S1.shape[0]) 
    else: 
        idx = np.random.randint(0, S1.shape[0], S1.shape[0]) 

    S1_avg = np.mean(S1[idx,:], axis=0) # average over trials 
    S2_avg = np.mean(S2[idx,:], axis=0) 
    
    return S1_avg, S2_avg 

def bootstrap(S1,S2): 
    
    S1_avg = [] 
    S2_avg = [] 
    
    for _ in range(gv.n_boot): 
        if gv.n_boot==1: 
            idx = np.arange(0, S1.shape[0]) 
        else: 
            idx = np.random.randint(0, S1.shape[0], S1.shape[0]) 
        S1_avg.append(np.mean(S1[idx,:], axis=0)) 
        S2_avg.append(np.mean(S2[idx,:], axis=0)) 
        
    S1_avg = np.asarray(S1_avg) 
    S2_avg = np.asarray(S2_avg) 

    return S1_avg, S2_avg 

def bootstrap_clf_par(X, y, clf): 

    if gv.n_boot==1: 
        print('no boot') 
        idx = np.arange(0, X.shape[0]) 
    else: 
        idx = np.random.randint(0, X.shape[0], X.shape[0]) 
        
    X_sample = X[idx] 
    y_sample = y[idx] 

    X_sample = StandardScaler().fit_transform(X_sample) 
    # scaler = StandardScaler().fit(X) 
    # X_sample = scaler.transform(X_sample) 
    
    clf.fit(X_sample, y_sample) 
    coefs_samples = clf.coef_.flatten() 
    
    return coefs_samples 

def bootstrap_clf(X, y, clf): 

    coefs_samples = [] 
    scaler = StandardScaler().fit(X) 
    for _ in range(gv.n_boot): 
        if gv.n_boot==1: 
            print('no boot') 
            idx = np.arange(0, X.shape[0]) 
        else: 
            idx = np.random.randint(0, X.shape[0], X.shape[0]) 
            
        X_sample = X[idx] 
        # X_sample = StandardScaler().fit_transform(X_sample) 
        X_sample = scaler.transform(X_sample) 
        y_sample = y[idx] 
        
        clf.fit(X_sample, y_sample) 
        coefs_samples.append( clf.coef_.flatten() ) 
        
    return coefs_samples

def unit_vector(vector): 
    """ Returns the unit vector of the vector.  """ 
    u = vector / np.linalg.norm(vector) 
    return u 

def angle_between(v1, v2): 
    """ Returns the angle in radians between vectors 'v1' and 'v2':: """ 
    v1_u = unit_vector(v1) 
    v2_u = unit_vector(v2)    
    # return np.arccos(np.dot(v1_u, v2_u)) 
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) 

def get_cos(coefs): 
    """ Returns the cosine of the angle alpha between vector coefs[0] (early delay) and coefs[1] (late delay) """
    alphas = [] 
    cos_alp=[] 
    for j in np.arange(0, coefs.shape[0]): 
        alpha = angle_between(coefs[0], coefs[j]) 
        alphas.append(alpha) 
        cos_alp.append(np.cos(alpha)) 
        
    return alphas, cos_alp 

def EDvsLD(X_proj, IF_CONCAT, IF_EDvsLD, IF_CLASSIFY, NO_PCA=0, C=1e0): 

    gv.n_boot = int(1e3) 
    num_cores = multiprocessing.cpu_count() 

    if IF_CONCAT:
        print('Concatenated Stim and ED')
    if IF_EDvsLD:
        print('angle btw ED and other epochs')
    if not IF_CLASSIFY:
        print('evaluate clusters mean with bootstrap') 
    if IF_CLASSIFY: 
        print('Use a linear classifier') 

    print('n_boot', gv.n_boot)
    # IF_CONCAT = 0 
    # IF_EDvsLD = 1 

    # IF_BOOTSTRAP = 0 
    # IF_CLASSIFY = 1
    
    IF_SHUFFLE = 0 

    if IF_EDvsLD:
        gv.epochs = ['ED', 'MD', 'LD'] 
    else:
        gv.epochs = ['Stim', 'ED', 'MD', 'LD']
        
    X_stim = np.mean(X_proj[:,:,:,:,gv.bins_stim],axis=-1) 
    X_ED = np.mean(X_proj[:,:,:,:,gv.bins_ED],axis=-1) 
    X_MD = np.mean(X_proj[:,:,:,:,gv.bins_MD],axis=-1) 
    X_LD = np.mean(X_proj[:,:,:,:,gv.bins_LD],axis=-1) 

    # print('Stim', X_stim.shape) 
    # print('ED', X_ED.shape) 
    # print('MD', X_MD.shape) 
    # print('LD', X_LD.shape) 
    
    if IF_CONCAT:
        X_stim_S1 = []
        X_stim_S2 = []

        X_ED_S1 = []
        X_ED_S2 = [] 
        for i in range(X_proj.shape[0]): 
            X_stim_S1.append(X_stim[i,0]) 
            X_stim_S2.append(X_stim[i,1]) 

            X_ED_S1.append(X_ED[i,0]) 
            X_ED_S2.append(X_ED[i,1]) 

        X_stim_S1 = np.vstack(np.asarray(X_stim_S1))
        X_stim_S2 = np.vstack(np.asarray(X_stim_S2))

        X_ED_S1 = np.vstack(np.asarray(X_ED_S1))
        X_ED_S2 = np.vstack(np.asarray(X_ED_S2))

    cos_samples_trials = [] 
    dist_samples_trials = [] 
    for n_trial, gv.trial in enumerate(gv.trials):

        if not IF_CONCAT: 
            X_stim_S1 = X_stim[n_trial,0] 
            X_stim_S2 = X_stim[n_trial,1] 

            X_ED_S1 = X_ED[n_trial,0]        
            X_ED_S2 = X_ED[n_trial,1] 

        X_MD_S1 = X_MD[n_trial,0] 
        X_MD_S2 = X_MD[n_trial,1] 

        # print(gv.trial, 'X_MD_S1', X_MD.shape) 
        
        X_LD_S1 = X_LD[n_trial,0] 
        X_LD_S2 = X_LD[n_trial,1] 

        # print(gv.trial, 'X_LD_S1', X_LD.shape) 
        
        if not IF_CLASSIFY:
            X_stim_S1_avg, X_stim_S2_avg = zip(*Parallel(n_jobs=num_cores)(delayed(bootstrap_par)(X_stim_S1, X_stim_S2) for i in range(gv.n_boot)))
            X_ED_S1_avg, X_ED_S2_avg = zip(*Parallel(n_jobs=num_cores)(delayed(bootstrap_par)(X_ED_S1, X_ED_S2) for i in range(gv.n_boot)))
            X_MD_S1_avg, X_MD_S2_avg = zip(*Parallel(n_jobs=num_cores)(delayed(bootstrap_par)(X_MD_S1, X_MD_S2) for i in range(gv.n_boot)))
            X_LD_S1_avg, X_LD_S2_avg = zip(*Parallel(n_jobs=num_cores)(delayed(bootstrap_par)(X_LD_S1, X_LD_S2) for i in range(gv.n_boot)))
            
            # X_stim_S1_avg, X_stim_S2_avg = bootstrap(X_stim_S1, X_stim_S2) 
            # X_ED_S1_avg, X_ED_S2_avg = bootstrap(X_ED_S1, X_ED_S2) 
            # X_MD_S1_avg, X_MD_S2_avg = bootstrap(X_MD_S1, X_MD_S2) 
            # X_LD_S1_avg, X_LD_S2_avg = bootstrap(X_LD_S1, X_LD_S2) 

            cos_samples = []
            dist_samples = []
            for i in range(gv.n_boot): 
                X_stim_avg = np.array([X_stim_S1_avg[i], X_stim_S2_avg[i]]) 
                dX_stim = X_stim_avg[0]-X_stim_avg[1] 

                dist_stim = distance.euclidean(X_stim_avg[0], X_stim_avg[1]) 
                
                X_ED_avg = [X_ED_S1_avg[i], X_ED_S2_avg[i]] 
                dX_ED = X_ED_avg[0]-X_ED_avg[1] 
                dist_ED = distance.euclidean(X_ED_avg[0], X_ED_avg[1]) 

                X_MD_avg = [X_MD_S1_avg[i], X_MD_S2_avg[i]] 
                dX_MD = X_MD_avg[0]-X_MD_avg[1] 
                dist_MD = distance.euclidean(X_MD_avg[0], X_MD_avg[1]) 
                
                X_LD_avg = [X_LD_S1_avg[i], X_LD_S2_avg[i]] 
                dX_LD = X_LD_avg[0]-X_LD_avg[1] 
                dist_LD = distance.euclidean(X_LD_avg[0], X_LD_avg[1]) 

                if IF_EDvsLD: 
                    dX = np.asarray([dX_ED, dX_MD, dX_LD])
                    dist = np.asarray([dist_ED, dist_MD, dist_LD])
                else:
                    dX = np.asarray([dX_stim, dX_ED, dX_MD, dX_LD])
                    dist = np.asarray([dist_stim, dist_ED, dist_MD, dist_LD])
                    
                alpha, cos_alp = get_cos(dX) 
                cos_samples.append(cos_alp) 
                dist_samples.append(dist)
                
            cos_samples_trials.append(cos_samples) 
            cos_samples = np.asarray(cos_samples) 
            # print('cos_samples', cos_samples.shape) 
            dist_samples_trials.append(dist_samples) 

            mean_dist = np.mean(dist_samples, axis=0)            
            q1_dist = mean_dist - np.percentile(dist_samples, 25, axis=0) 
            q3_dist = np.percentile(dist_samples, 75, axis=0) - mean_dist 
            

            print('trial', gv.trial, 'dist', mean_dist, 'q1', q1_dist, 'q3', q3_dist) 
            
            mean_cos = np.mean(cos_samples, axis=0) 
            q1 = mean_cos - np.percentile(cos_samples, 25, axis=0) 
            q3 = np.percentile(cos_samples, 75, axis=0) - mean_cos

            print('trial', gv.trial, 'cos', mean_cos, 'q1', q1, 'q3', q3) 
            plot.plot_cosine_bars(mean_cos, [], q1, q3, IF_SHUFFLE=IF_SHUFFLE) 

        if IF_CLASSIFY: 
            coefs = [] 
            X_stim_S1_S2 = np.vstack([X_stim_S1, X_stim_S2]) 
            X_ED_S1_S2 = np.vstack([X_ED_S1, X_ED_S2]) 
            X_MD_S1_S2 = np.vstack([X_MD_S1, X_MD_S2]) 
            X_LD_S1_S2 = np.vstack([X_LD_S1, X_LD_S2]) 
            
            if IF_EDvsLD: 
                X_epochs = [X_ED_S1_S2 , X_MD_S1_S2 , X_LD_S1_S2 ] 
            else: 
                X_epochs = [X_stim_S1_S2, X_ED_S1_S2 , X_MD_S1_S2 , X_LD_S1_S2 ] 
            
            for X in X_epochs: 
                y = np.array([np.zeros(int(X.shape[0]/2)), np.ones(int(X.shape[0]/2))]).flatten() 
                # print(X.shape, y.shape) 
                
                # C=1e-6 
                penalty='l2' 
                solver = 'liblinear' 
                clf = LogisticRegression(C=C, solver=solver, penalty=penalty,tol=1e-6, max_iter=int(1e6), fit_intercept=True) 
                # clf = svm.LinearSVC(C=C, penalty=penalty, loss='squared_hinge', dual=False, tol=1e-6, max_iter=int(1e6), fit_intercept=False) 
                # clf = svm.LinearSVR(C=C, tol=1e-6, max_iter=int(1e6))
                # clf = LinearDiscriminantAnalysis()
                
                # X = StandardScaler().fit_transform(X) 
                # clf.fit(X, y)
                # coefs.append( clf.coef_.flatten() )
                boot_coefs = Parallel(n_jobs=num_cores)(delayed(bootstrap_clf_par)(X, y, clf) for i in range(gv.n_boot)) 
                coefs.append(boot_coefs) 
                
            # coefs = np.vstack(np.asarray(coefs)).reshape(len(X_epochs), int(gv.n_boot * X.shape[0]), X.shape[1]) 
            coefs = np.vstack(np.asarray(coefs)).reshape(len(X_epochs), int(gv.n_boot), X.shape[1]) 
            # coefs = np.asarray(coefs) 
            # print(coefs.shape) 
            
            cos_samples = [] 
            for boot_sample in range(coefs.shape[1]): 
                alpha, cos_alp = get_cos(coefs[:,boot_sample,:]) 
                cos_samples.append(cos_alp)
                
            cos_samples_trials.append(cos_samples)
            cos_samples = np.asarray(cos_samples) 
            # print(cos_samples.shape)
            
            mean_cos = np.mean(cos_samples, axis=0) 
            # print(cos_alp.shape) 
            q1 = mean_cos - np.percentile(cos_samples, 25, axis=0) 
            q3 = np.percentile(cos_samples, 75, axis=0) - mean_cos 
            
            print('trial', gv.trial, 'cos', mean_cos, 'q1', q1, 'q3', q3) 
            plot.plot_cosine_bars(mean_cos, [], q1, q3) 
        
            # alpha, cos_alp = get_cos(coefs) 
            # print('trial', gv.trial, 'cos', cos_alp) 
            # plot.plot_cosine_bars(cos_alp) 

    cos_samples_trials = np.asarray(cos_samples_trials)
    print(cos_samples_trials.shape)

    cols = [-4/10, -1/10, 2/10]
    high = [1.3, 1.1] 

    p_values = []
    for i in range(1, cos_samples_trials.shape[2]): # epochs
        for j in range(1, cos_samples_trials.shape[0]): # trials 
            sample_1  = cos_samples_trials[0,:,i] 
            sample_2  = cos_samples_trials[j,:,i]
            t_score, p_value = stats.ttest_ind(sample_1, sample_2, equal_var=False)
            if t_score>0:
                p_value = p_value/2
            else:
                p_value = 1-p_value/2
            p_values.append(p_value) 
            print(gv.epochs[i], 'ND vs', gv.trials[j], 't_score', t_score, 'p_value', p_value) 
            
    p_values = np.asarray(p_values).reshape(len(gv.epochs)-1, 2)
    print(p_values.shape)
    print(p_values)

    cols = [-4/10, -1/10, 2/10] 
    high = [1.3, 1.1] 
    for i in range(0,len(gv.epochs)-1): 
        for j in range(1, len(cols)): 
            plt.plot( [i+cols[0], i+cols[j]] , [high[j-1], high[j-1]] , lw=.8, c='k') 
            if p_values[i,j-1]<=.001: 
                plt.text((2*i+cols[0]+cols[j])*.5, high[j-1]-.1, "***", ha='center', va='bottom', color='k', fontsize=8) 
            elif p_values[i,j-1]<=.01:
                plt.text((2*i+cols[0]+cols[j])*.5, high[j-1]-.1, "**", ha='center', va='bottom', color='k', fontsize=8) 
            elif p_values[i,j-1]<=.05:
                plt.text((2*i+cols[0]+cols[j])*.5, high[j-1]-.1, "*", ha='center', va='bottom', color='k', fontsize=8)
            elif p_values[i,j-1]>.05:
                plt.text((2*i+cols[0]+cols[j])*.5, high[j-1], "ns", ha='center', va='bottom', color='k', fontsize=6)

    plt.ylim([-1, 1.5]) 
    figdir = plot.figDir() 
    if gv.laser_on: 
        figdir = figdir + '/laser_on/'
        figtitle = '%s_%s_cos_alpha_pca_laser_on' % (gv.mouse, gv.session) 
    else: 
        figtitle = '%s_%s_cos_alpha_pca_laser_off' % (gv.mouse, gv.session) 

    if NO_PCA: 
        figdir = figdir + '/no_pca/'
    
    if IF_CLASSIFY:
        figdir = figdir + '/clf/'
        clf_name = clf.__class__.__name__ 
        
        if(clf_name == 'LogisticRegression'): 
            clf_param = '/C_%.3f_penalty_%s_solver_%s/' % (C, penalty, solver)
            figdir = figdir + clf_name + clf_param 
        else:
            clf_param = '/C_%.3f/' % C 
            figdir = figdir + clf_name + clf_param 

    if IF_EDvsLD: 
        figdir = figdir + '/EDvsLD/'
        
    if IF_CONCAT:
        figdir = figdir + '/concat_stim_ED/'

    if not os.path.isdir(figdir):
        os.makedirs(figdir)
        
    # plot.save_fig(figtitle, figdir) 
