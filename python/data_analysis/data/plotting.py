from .libs import *
from . import constants as gv

shade_alpha = 0.2
lines_alpha = 0.8
pal = ['r','b','y']

def figDir():
    script_dir = os.path.dirname(__file__) 
    figdir = script_dir + '/figs/' % gv.n_components 

    if not os.path.isdir(figdir):
        os.makedirs(figdir)

    return figdir

def add_stim_to_plot(ax, bin_start=0):

    ax.axvspan(gv.bins_ED[0]-bin_start, gv.bins_ED[-1]-bin_start, alpha=shade_alpha, color='gray')
    ax.axvspan(gv.bins_MD[0]-bin_start, gv.bins_MD[-1]-bin_start, alpha=shade_alpha, color='blue')    
    ax.axvspan(gv.bins_LD[0]-bin_start, gv.bins_LD[-1]-bin_start, alpha=shade_alpha, color='gray')
    
    ax.axvline(gv.bins_dist[0]-bin_start, alpha=lines_alpha, color='k', ls='-')
    ax.axvline(gv.bins_dist[-1]-bin_start, alpha=lines_alpha, color='k', ls='-')

    ax.axvline(gv.bins_cue[0]-bin_start, alpha=lines_alpha, color='k', ls='--')
    ax.axvline(gv.bins_cue[-1]-bin_start, alpha=lines_alpha, color='k', ls='--')

def add_orientation_legend(ax):
    custom_lines = [Line2D([0], [0], color=pal[k], lw=4) for
                    k in range(len(gv.trials))]
    labels = [t for t in gv.trials]
    ax.legend(custom_lines, labels,
              frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.9,1])    

def plot_cosine_bars(cos_alp, mean_cos=[], q1=[], q3=[], IF_SHUFFLE=0):

    # gv.epochs = ['ED','MD','LD']
    
    if len(q1)<2:
        yerr = np.zeros(len(gv.epochs)-1) 
    else:
        yerr=np.array([q1[1:],q3[1:]]) 
    
    if gv.laser_on:
        figtitle = '%s_%s_cos_alpha_pca_laser_on' % (gv.mouse, gv.session)
    else:
        figtitle = '%s_%s_cos_alpha_pca_laser_off' % (gv.mouse, gv.session)

    ax = plt.figure(figtitle).add_subplot()
    xticks = np.arange(0,len(gv.epochs)-1) 

    # print(xticks.shape, cos_alp[1:].shape,yerr.shape)

    if IF_SHUFFLE: 
        width = 1/10. 
    else: 
        width = 2/10. 
        
    if('ND' in gv.trial): 
        ax.bar(xticks - 4/10, cos_alp[1:], width, yerr=yerr ,label=gv.trial, color='r') ; 
        if IF_SHUFFLE: 
            ax.bar(xticks - 3/10, mean_cos[1:], width, yerr=yerr, color='r', alpha=0.5) ; 
        
    if('D1' in gv.trial): 
        ax.bar(xticks - 1/10, cos_alp[1:], width, yerr=yerr,label=gv.trial, color='b') ; 
        if IF_SHUFFLE:
            ax.bar(xticks, mean_cos[1:], width, yerr=yerr, color='b', alpha=0.5) ; 
                
    if('D2' in gv.trial):
        ax.bar(xticks + 2/10 , cos_alp[1:], width, yerr=yerr, label=gv.trial, color='y') ; 
        if IF_SHUFFLE:
            ax.bar(xticks + 3/10, mean_cos[1:], width, yerr=yerr, color='y', alpha=0.5) ;
                    
    plt.ylabel('cos($\\alpha$)') 
    plt.xlabel('epochs') 
    labels = gv.epochs ;
    ax.set_xticks(xticks) ; 
    ax.set_xticklabels(labels[1:]) ; 
    ax.set_ylim([-1,1]) 
    # ax.legend()

def save_fig(figname, figdir):
    plt.figure(figname) 
    plt.savefig(figdir + figname +'.svg',format='svg') 
    print('saved to', figdir + figname + '.svg')
