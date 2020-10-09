from std_lib import *
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis')

import data.constants as gv
importlib.reload(gv) ;

import data.utils as data
importlib.reload(data) ;

from sklearn_lib import *
from mne_lib import *

import utils as decode 
importlib.reload(decode) ; 

import data.fct_facilities as fac
importlib.reload(fac) ;
fac.SetPlotParams()

IF_EPOCHS = 0

if not IF_EPOCHS:
    gv.epochs = ['all']

script_dir = os.path.dirname(__file__) 
figs_dir = os.path.join(script_dir, 'figs/') 
data_dir = os.path.join(script_dir, 'dat/')

C=1
penalty='l1'
solver='liblinear' # lbfgs
loss = 'squared_hinge'
dual=True
cv = 5 

# clf = LogisticRegressionCV(penalty=penalty, solver=solver, cv=cv) 
clf = LogisticRegression(C=C, penalty=penalty, solver=solver) 
# clf = svm.LinearSVC(C=1, penalty='l1', loss='squared_hinge', dual=False) 
# clf = LinearDiscriminantAnalysis() 

clf_name = clf.__class__.__name__

if(clf_name == 'LogisticRegression'):
    clf_param = '/C_%.1f_penalty_%s_solver_%s/' % (C, penalty, solver) 
    figs_dir = os.path.join(figs_dir, clf_name + clf_param) 
    data_dir = os.path.join(data_dir, clf_name + clf_param) 
else:
    figs_dir = os.path.join(figs_dir, clf_name) 
    data_dir = os.path.join(data_dir, clf_name) 

# if gv.laser_on: 
#     figs_dir = os.path.join(figs_dir, 'laser_on/') 
#     data_dir = os.path.join(data_dir, 'laser_on/') 

if IF_EPOCHS: 
    figs_dir = os.path.join(figs_dir, 'epochs/') 
    data_dir = os.path.join(data_dir, 'epochs/') 

figs_dir = os.path.join(figs_dir, 'mne/') 
data_dir = os.path.join(data_dir, 'mne/') 

if not os.path.isdir(figs_dir):
    os.makedirs(figs_dir)
    
if not os.path.isdir(data_dir):
    os.makedirs(data_dir) 

for gv.mouse in gv.mice : 

    data.get_sessions_mouse()
    data.get_stimuli_times()
    data.get_delays_times() 

    for gv.session in [gv.sessions[-1]] : 
        X_data, y_labels = data.get_fluo_data() 
        print('mouse', gv.mouse, 'session', gv.session, 'data X', X_data.shape,'y', y_labels.shape) 
        
        data.get_delays_times() 
        data.get_frame_rate() 
        data.get_bins(t_start=0) 
        
        gv.duration = X_data.shape[2]/gv.frame_rate 
        time = np.linspace(0, gv.duration, X_data.shape[2]) ; 
        
        for gv.trial in gv.trials : 
            X_S1_trials, X_S2_trials = data.get_S1_S2_trials(X_data, y_labels) 
            X_trials, y_trials = data.get_X_y_epochs(X_S1_trials, X_S2_trials) 
            
            print('trial:', gv.trial, 'X', X_trials.shape,'y', y_trials.shape) 
            
            scores, scores_std = decode.mne_cross_temp_clf(X_trials, y_trials, cv=cv) 
            
            decode.cross_temp_plot_mat(scores, IF_EPOCHS) 
            
            figname = '%s_session_%s_trial_%s_cross_temp_decoder' % (gv.mouse,gv.session,gv.trial) 
            plt.figure(figname)
            plt.savefig(figs_dir + figname +'.svg',format='svg')
