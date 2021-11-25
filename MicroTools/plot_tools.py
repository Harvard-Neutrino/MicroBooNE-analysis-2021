import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.pyplot import cm
import matplotlib.patches as mpatches

import seaborn as sns

fsize=10
rcparams={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
                'figure.figsize':(1.2*3.7,1.3*2.3617), 
                'legend.frameon': False,
                'legend.loc': 'best'  }
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
rc('text', usetex=True)
rc('font',**{'family':'serif', 'serif': ['Computer Modern Roman']})
rcParams.update(rcparams)
matplotlib.rcParams['hatch.linewidth'] = 0.3

axes_form  =[0.16,0.16,0.81,0.76]

def std_fig(ax_form=axes_form, 
            fig_size=(1.2*3.7,1.3*2.3617),
            rasterized=False):
    fig = plt.figure()
    ax = fig.add_axes(ax_form, rasterized=rasterized)
    ax.patch.set_alpha(0.0)
    return fig,ax

def double_axes_fig(height = 0.5,
                    gap = 0.1,
                    axis_bottom = [0.14,0.1,0.80,0.18], 
                    fig_size=(1.2*3.7,1.3*2.3617),
                    rasterized=False):

    fig = plt.figure(figsize=fig_size)
    axis_bottom = axis_bottom
    axis_bottom[-1] = height
    axis_top = axis_bottom+np.array([0, height+gap, 0, 1 - 2*height - gap - axis_bottom[1] - 0.05])
    ax1 = fig.add_axes(axis_top, rasterized=rasterized)
    ax2 = fig.add_axes(axis_bottom, rasterized=rasterized)
    ax1.patch.set_alpha(0.0)
    ax2.patch.set_alpha(0.0)
    # ax1.set_xticklabels([])
    return fig, ax1, ax2

def data_plot(ax, X, Y, xerr, yerr, zorder=2, label='data'):
    return ax.errorbar(X, Y, yerr= yerr, xerr = xerr, \
                    marker="o", markeredgewidth=0.5, capsize=1.0,markerfacecolor="black",\
                    markeredgecolor="black",ms=2,  lw = 0.0, elinewidth=0.8,
                    color='black', label=label, zorder=zorder)

def step_plot(ax, x, y, lw=1, color='red', label='signal', where = 'post', dashes=(3,0), zorder=3):
    return ax.step( np.append(x, np.max(x)+x[-1]),
                    np.append(y, 0.0),
                    where=where,
                    lw = lw, 
                    dashes=dashes,
                    color = color, 
                    label = label, zorder=zorder)


def plot_MB_vertical_region(ax, color='dodgerblue', label=r'MiniBooNE $1 \sigma$'):
    ##########
    # MINIBOONE 2018
    matplotlib.rcParams['hatch.linewidth'] = 0.7
    y = [0,1e10]
    NEVENTS=381.2
    ERROR = 85.2
    xleft = (NEVENTS-ERROR)/NEVENTS
    xright = (NEVENTS+ERROR)/NEVENTS
    ax.fill_betweenx(y,[xleft,xleft],[xright,xright], 
                        zorder=3,
                        ec=color, fc='None',
                        hatch='\\\\\\\\\\',
                        lw=0,
                        label=label)

    ax.vlines(1,0,1e10, zorder=3, lw=1, color=color)
    ax.vlines(xleft,0,1e10, zorder=3, lw=0.5, color=color)
    ax.vlines(xright,0,1e10, zorder=3, lw=0.5, color=color)
