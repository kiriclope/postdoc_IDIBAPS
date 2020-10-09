from std_lib import * 
from sklearn_lib import * 
from mne_lib import * 

sys.path.append('../../') 
import data.constants as gv
import data.utils as func 

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    u = vector / np.linalg.norm(vector)
    return u 

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2':: """
    v1_u = unit_vector(v1) 
    v2_u = unit_vector(v2) 
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) 

def pca(X_trials):
    X_red = []
    for trial in range(0,X_trials.shape[0]) :
        X = X_trials[trial,:,:]
        X = StandardScaler().fit_transform(X)
        X_red.append(PCA().fit_transform(X))
    X_red = np.asarray(X_red)
    return X_red

def coefs_clf(X, y, clf=LogisticRegression(), shuffle=0): 
    
    Pipe = make_pipeline(StandardScaler(), LinearModel(clf))
    
    time_decod = SlidingEstimator(Pipe, n_jobs=1, scoring='accuracy', verbose=False)
    time_decod.fit(X, y)
    
    coefs = get_coef(time_decod, inverse_transform=False) 
    coefs = np.asarray(coefs).transpose() 
    # print(coefs.shape)
    return coefs

def grid_search_cv_clf(X_trials, y_trials, shuffle=0, cv=5): 

    clf = LogisticRegression()
    
    pipe = Pipeline([('scale', StandardScaler()), ('classifier', clf)])
    
    param_grid = [{'classifier': [clf],
                   'classifier__penalty' : ['l1'],
                   'classifier__C' : [1] , # np.logspace(-1, 1, 100),
                   'classifier__solver' : ['liblinear'],
                   'classifier__multi_class' : ['ovr']}
    ]

    coefs = [] 
    coefs_CV = [] 
    
    for bin in np.arange(0,X_trials.shape[2]): # epochs
        X = X_trials[:,:,bin] 
        # X = StandardScaler().fit_transform(X)
        
        if bin==0: 
            y = y_trials 
            if shuffle: 
                random.shuffle(y) 

        search = GridSearchCV(pipe, param_grid=param_grid, cv=5, verbose=False, n_jobs=-1) 
        
        best_model = search.fit(X,y) 
        best_penalty = best_model.best_estimator_.get_params()['classifier__penalty'] 
        best_C = best_model.best_estimator_.get_params()['classifier__C'] 
        best_solver = best_model.best_estimator_.get_params()['classifier__solver'] 

        print('trial', gv.trial, 'epoch', gv.epochs[bin], 'C', best_C, 'penalty', best_penalty, 'solver', best_solver) 

        best_clf = best_model.best_estimator_.named_steps['classifier']
        coef_array = best_clf.coef_ 
        coefs.append( coef_array.flatten() ) 
   
    coefs = np.asarray(coefs).transpose()
    return coefs, best_clf

def get_cos(coefs): 
    """ Returns the cosine of the angle alpha between vector coefs[0] (early delay) and coefs[1] (late delay) """
    alphas = []
    cos_alp=[]
    # for i in np.arange(0,coefs.shape[0]):
    #     for j in np.arange(i+1,coefs.shape[0]):
    #         alpha = angle_between(coefs[i], coefs[j])
    #         alphas.append(alpha) 
    #         cos_alp.append(np.cos(alpha)) 
    # print(cos_alp)
    for j in np.arange(0, coefs.shape[0]): 
        alpha = angle_between(coefs[0], coefs[j]) 
        alphas.append(alpha) 
        cos_alp.append(abs(np.cos(alpha)))
    # print(cos_alp) 
    
    return alphas, cos_alp

def plot_cos_bar(mean_cos, std_cos):
    """ bar plot of the value of the cosine of alpha """
    labels = gv.trials ;
    xticks = np.arange(0,len(labels))
    width = .3
    
    figtitle = '%s_%s_cos_alpha' % (gv.mouse, gv.session)
    # ax = plt.figure(figtitle).add_subplot() 
    ax = plt.figure(figtitle).subplots() 
    rects = ax.bar(xticks+gv.dum*width/2, mean_cos, width, label=gv.mouse, yerr=std_cos) ; 
    # ax.legend() 
    ax.set_xticks(xticks) ;
    ax.set_xticklabels(labels) ; 
    
    plt.xlabel('trials')
    plt.ylabel('cos($\\alpha$) late vs early')

def get_z_score_cos_alp(cos_alp, mean_cos, std_cos):
    """ bar plot of the z-score : (observation - mean shuffle)/std_shuffle of the cosine of alpha """

    if len(mean_cos)==1:
        z_score_cos_alp = [ (cos_alp[i] - mean_cos[0]) / std_cos[0] for i in range(0,3) ]
    else:
        z_score_cos_alp = [ (cos_alp[i] - mean_cos[i]) / std_cos[i] for i in range(0,3) ]
        
    print(z_score_cos_alp)

    labels = gv.trials ;
    xticks = np.arange(len(labels))
    width = .3
    
    # figtitle = '%s_%s_z_score' % (gv.mouse, gv.session)
    # ax = plt.figure(figtitle).add_subplot()

    # ax.bar(xticks, z_score_cos_alp, width, label=gv.mouse) ;
    # ax.set_xticks(xticks) ;
    # ax.set_xticklabels(labels) ;
    # ax.legend(loc='best')

    # plt.xlabel('trials')
    # plt.ylabel('cos($\\alpha$)')
    # plt.ylabel('z-score of cos($\\alpha$)')

    return z_score_cos_alp

def get_z_score_alpha(alpha, mean_alpha, std_alpha):
    """ bar plot of the z-score : (observation - mean shuffle)/std_shuffle of alpha """
    z_score_alpha = [ (alpha[i] - mean_alpha[i]) / std_alpha[i] for i in range(0,3) ]
    print(z_score_alpha)

    labels = gv.trials ;
    xticks = np.arange(len(labels))
    width = .3
    
    figtitle = '%s_%s_z_score' % (gv.mouse, gv.session)
    ax = plt.figure(figtitle).add_subplot()

    ax.bar(xticks, z_score_alpha, width, label=gv.mouse, color='b') ; 
    ax.set_xticks(xticks) ;
    ax.set_xticklabels(labels) ;
    ax.legend(loc='best')

    plt.xlabel('trials')
    plt.ylabel('z-score')

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

def get_p_value_alp(z_score_alp):
    """ bar plot of the z-score : (observation - mean shuffle)/std_shuffle of alpha """
    p_value = []
    for i in range(0,len(z_score_alp)): 
        if z_score_alp[i]<0:
            p_value.append(st.norm.cdf(z_score_alp[i]))
        else:
            p_value.append(1-st.norm.cdf(z_score_alp[i]))

    print('p_value', p_value) 

    # labels = gv.trials ;
    # xticks = np.arange(len(labels))
    # width = .3
    
    # figtitle = '%s_%s_p_value' % (gv.mouse, gv.session)
    # ax = plt.figure(figtitle).add_subplot()

    # ax.bar(xticks, p_value, width, label=gv.mouse, color='b') ; 
    # ax.set_xticks(xticks) ;
    # ax.set_xticklabels(labels) ;
    # ax.legend(loc='best')

    # plt.xlabel('trials')
    # plt.ylabel('p_value')

    return p_value

def create_dir(clf):
    figs_dir = os.path.join(figs_dir, clf.__class__.__name__+'/') 
    clf_name = clf.__class__.__name__ 

    if(clf_name == 'LogisticRegression'): 
        clf_param = '/C_%.1f_penalty_%s_solver_%s/' % (C, penalty, solver)
        figs_dir = os.path.join(figs_dir, clf_name + clf_param) 
        data_dir = os.path.join(data_dir, clf_name + clf_param) 
    else:
        figs_dir = os.path.join(figs_dir, clf_name) 
        data_dir = os.path.join(data_dir, clf_name) 

    if gv.laser_on: 
        figs_dir = os.path.join(figs_dir, 'laser_on/') 
        data_dir = os.path.join(data_dir, 'laser_on/') 

    if IF_CV: 
        figs_dir = os.path.join(figs_dir, 'crossval/') 
        data_dir = os.path.join(data_dir, 'crossval/') 

    if IF_GRID: 
        figs_dir = os.path.join(figs_dir, 'grid_search_cv/') 
        data_dir = os.path.join(data_dir, 'grid_search_cv/') 

    if IF_DETREND :
        figs_dir = os.path.join(figs_dir, 'detrend/') 
        data_dir = os.path.join(data_dir, 'detrend/') 
        if IF_POLY :
            figs_dir = os.path.join(figs_dir, 'detrend/poly/') 
            data_dir = os.path.join(data_dir, 'detrend/poly/') 

    if not os.path.isdir(figs_dir):
        os.makedirs(figs_dir)
    
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir) 

def detrend_data(X_trial, poly_fit=1, degree=9): 
    """ Detrending of the data, if poly_fit=1 uses polynomial fit else linear fit. """
    # X_trial : # neurons, # times 
    
    model = LinearRegression()
    fit_values_trial = []
    
    for i in range(0, X_trial.shape[0]): # neurons
        indexes = range(0, X_trial.shape[1]) # neuron index
        values = X_trial[i] # fluo value 
        
        indexes = np.reshape(indexes, (len(indexes), 1))
        
        if poly_fit:
            poly = PolynomialFeatures(degree=degree) 
            indexes = poly.fit_transform(indexes)
            
        model.fit(indexes, values)
        fit_values = model.predict(indexes) 
        
        fit_values_trial.append(fit_values) 
        
    fit_values_trial = np.asarray(fit_values_trial)
    return fit_values_trial 
