from libs import *
sys.path.append('../../')
import data.constants as gv

shade_alpha      = 0.2
lines_alpha      = 0.8
pal = ['r','b','y']

def add_stim_to_plot(ax):
    ax.axvspan(gv.t_early_delay[0], gv.t_early_delay[1], alpha=shade_alpha,
               color='gray')

    ax.axvspan(gv.t_late_delay[0], gv.t_late_delay[1], alpha=shade_alpha,
               color='gray')
    
    ax.axvspan(gv.t_DRT_delay[0], gv.t_DRT_delay[1], alpha=shade_alpha,
               color='blue')
    
    ax.axvline(gv.t_sample[0], alpha=lines_alpha, color='gray', ls='--')
    ax.axvline(gv.t_sample[1], alpha=lines_alpha, color='gray', ls='--')

    ax.axvline(gv.t_early_delay[0], alpha=lines_alpha, color='gray', ls='--')
    ax.axvline(gv.t_early_delay[1], alpha=lines_alpha, color='gray', ls='--')

    ax.axvline(gv.t_DRT_delay[0], alpha=lines_alpha, color='gray', ls='--')
    ax.axvline(gv.t_DRT_delay[1], alpha=lines_alpha, color='gray', ls='--')

    ax.axvline(gv.t_late_delay[0], alpha=lines_alpha, color='gray', ls='--')
    ax.axvline(gv.t_late_delay[1], alpha=lines_alpha, color='gray', ls='--')

    ax.axvline(gv.t_distractor[0], alpha=lines_alpha, color='gray', ls='--')
    ax.axvline(gv.t_distractor[1], alpha=lines_alpha, color='gray', ls='--')

    # ax.axvline(gv.t_cue[0], alpha=lines_alpha, color='gray', ls='--')
    # ax.axvline(gv.t_cue[1], alpha=lines_alpha, color='gray', ls='--')

def add_orientation_legend(ax):
    custom_lines = [Line2D([0], [0], color=pal[k], lw=4) for
                    k in range(len(gv.trials))]
    labels = ['{}$^\circ$'.format(t) for t in gv.trials]
    ax.legend(custom_lines, labels,
              frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.9,1])    
