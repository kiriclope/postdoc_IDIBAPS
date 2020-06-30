import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn import svm
from sklearn.preprocessing import StandardScaler
import sklearn.ensemble
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import cross_validate


def cross_validate_clf(split_windows, y_S1_S2, cv, scoring, clf):
    S2_f_score = []
    S1_f_score = []

    S2_f_score_std = []
    S1_f_score_std = []

    j = 0
    for X_window in split_windows:
        j = j+1
        # print('Window iteration #' + str(j))
        # print('window shape of current iteration')
        # print(X_window.shape)
        #reshaping the array for samples = trials features = electrodes by time features
        X_window  = np.reshape((X_window), (X_window.shape[0], X_window.shape[1]* X_window.shape[2]))
        # X_window = np.mean(X_window, axis=1)
        X_window = StandardScaler().fit_transform(X_window)
        #scikit learn function to crossvalidate
        #array of the form samples x features 2D = X_window
        out = cross_validate(clf, X_window, y_S1_S2, scoring=scoring, cv=cv, return_train_score=False, n_jobs=-1)
        i = 0
        num_elements_cv = len(out)
        #iterating over the dictionary
        for key, value in out.items():
            i = i + 1
            # print('crossvalidation  value #' + str(i))
            # print(key)
            # print(value)
            if i == num_elements_cv-1:
                #calculating the mean of the crossvalidation results
                S1_f_score.append(np.average(value))
                #calculating the standard deviation of the crossvalidated results
                S1_f_score_std.append(np.std(value))
    #             print('S1_f_score')
    #             print(S1_f_score)
            if i == num_elements_cv:
                S2_f_score.append(np.average(value))
                S2_f_score_std.append(np.std(value))
    #             print('S2_f_score')
    #             print(S2_f_score)

    #transforming lists to numpy arrays
    S1_f_score = np.array([S1_f_score])
    S1_f_score = S1_f_score[0]

    S2_f_score = np.array([S2_f_score])
    S2_f_score = S2_f_score[0]

    S1_f_score_std = np.array([S1_f_score_std])
    S1_f_score_std = S1_f_score_std[0]

    S2_f_score_std = np.array([S2_f_score_std])
    S2_f_score_std = S2_f_score_std[0]

    # #saving the results of the current modelling for future use (and not need to re run again)
    # np.savetxt(folder + area + '_S1_score_' + str(time_window) + 'ms_window.txt', S1_f_score, fmt='%10.5f')
    # np.savetxt(folder + area + '_S1_score_std' + str(time_window) + 'ms_window.txt', S1_f_score_std, fmt='%10.5f')

    # np.savetxt(folder + area + '_S2_score_' + str(time_window) + 'ms_window.txt', S2_f_score, fmt='%10.5f')
    # np.savetxt(folder + area + '_S2_score_std' + str(time_window) + 'ms_window.txt', S2_f_score_std, fmt='%10.5f')

    return S1_f_score, S2_f_score, S1_f_score_std, S2_f_score_std

###############################################################################
###############################################################################

def plot_decoding_results(S1_f_score, S2_f_score, S1_f_score_std, S2_f_score_std,time):
    plt.plot(time, S1_f_score, label='S1', marker='.', color='green')
    plt.fill_between(time, S1_f_score - S1_f_score_std, S1_f_score + S1_f_score_std, alpha=0.25, color='green')
    plt.plot(time, S2_f_score, label='S2', marker='.', color='magenta')
    plt.fill_between(time, S2_f_score - S2_f_score_std, S2_f_score + S2_f_score_std, alpha=0.25, color='magenta')


    # horizontal line to exemplify our chance level
    y_for_chance = np.repeat(0.50, len(S1_f_score) )

    plt.plot(time, y_for_chance, '--', c='black')
    plt.ylim([0, 1])
    plt.grid(which='both')
    plt.minorticks_on()
    plt.xlabel('Time (ms)')
    plt.legend(loc='upper right')
    plt.ylabel('decoder f-score')

    # show y axis as percentage
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    plt.axvline(x=2, c='black', linestyle='dashed')
    plt.axvline(x=3, c='black', linestyle='dashed')

    # vertical lines for cue and stimulus
    plt.text(2.1, 0.95, 'Sample', rotation=0)
    plt.text(9.1, 0.95, 'Test', rotation=0)

    plt.axvline(x=9, c='black', linestyle='dashed')
    plt.axvline(x=10, c='black', linestyle='dashed')
    plt.title('S1 and S2 decoding')
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    # fig.savefig(folder + area + '_S1_S2_plot.png', format='png')
