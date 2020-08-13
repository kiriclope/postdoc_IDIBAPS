from cross_temp.std_lib import *
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis')

import data.global_vars as gv
import data.utils as data
importlib.reload(sys.modules['data.utils']) ;

import cross_temp.utils as decode
from cross_temp.sklearn_lib import *
from cross_temp.mne_lib import *
importlib.reload(sys.modules['cross_temp.utils']) ;

for gv.folder in ['C57_2_DualTask', 'ChRM04', 'JawsM15'] :
    print(gv.folder)
    data.get_sessions_folder()
    for gv.session in gv.sessions :
        print(gv.session)
        X_data, y_labels = data.get_fluo_data()

        data.get_delays_times()
        data.get_frame_rate()
        gv.duration = X_data.shape[2]/gv.frame_rate
        time = np.linspace(0, gv.duration, X_data.shape[2]);

        for gv.trial in gv.trials :
            print(gv.trial)
            gv.n_bin = 1.5 * gv.frame_rate
            
            X_trials, y_trials = data.get_X_y_trials(X_data, y_labels) 
            scores, scores_std = decode.cross_temp_clf(X_trials, y_trials) 
            
            decode.cross_temp_plot_mat(scores) 
            # save_fig(folder, session, trial) 
            plt.close()

