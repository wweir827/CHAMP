import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patch
from matplotlib.path import Path


def plot_2d_domains(ind_2_domains, ax=None, col=None, close=False, widths=None, label=False):
    '''

    :param ind_2_domains:
    :param ax:
    :param col:
    :param close:
    :param widths:
    :param label:
    :return:

    '''
    if ax==None:
        f=plt.figure()
        ax=f.add_subplot(111)
    if widths==None:
        widths = np.random.sample(len(ind_2_domains))*3 + 1
    if col==None:
        cnorm=mcolors.Normalize(vmin=0,vmax=len(ind_2_domains))
        cmap=cm.get_cmap("Set1")
        colors=map(lambda(i): cmap(cnorm(i)),range(len(ind_2_domains)))
    i=0

    for i,ind_pts in enumerate(ind_2_domains.items()):
        ind,pts=ind_pts
        if hasattr(col,"__iter__" ):
            assert len(col) == len(ind_2_domains)
            c=col[i] #must match length
        else:
            c=colors[i] if col==None else col
        polypts=[(pt[0],pt[1]) for pt in pts]
        polypts+=[(pts[0][0],pts[0][1])]
        polycodes=[Path.MOVETO]+[Path.LINETO] *(len(polypts)-2) +[Path.CLOSEPOLY]
        polypath=Path(polypts,polycodes)
        polypatch = patch.PathPatch(polypath, facecolor=c, lw=2,alpha=.75)
        ax.add_patch(polypatch)
        xcrds=[x[0] for x in  pts  ]
        ycrds =[x[1] for x in pts ]
        ax.scatter(xcrds, ycrds, marker='x', c=c)
        # for i,x in enumerate(xcrds):
        #     jitter=np.random.uniform(-1,1)/10
        #     ax.text(x+jitter,ycrds[i]+jitter,i)
        i+=1
    return ax

def plot_single_layer_modularity(ind_2_domains,ax=None, col=None):
    '''
    Plot the piece-wise linear curve for CHAMP of single layer partitions
    :param ind_2_domains:
    :param ax: Matplotlib Axes object to draw the graph on
    :param col: Either a single color or list of colors with same length as number of domains
    :return :  ax  Reference to the ax

    '''
    if ax==None:
        f=plt.figure()
        ax=f.add_subplot(111)

    if col==None:
        cnorm=mcolors.Normalize(vmin=0,vmax=len(ind_2_domains))
        cmap=cm.get_cmap("Set1")
        colors=map(lambda(i): cmap(cnorm(i)),range(len(ind_2_domains)))
    i=0

    for i,ind_pts in enumerate(ind_2_domains.items()):
        ind, pts = ind_pts
        if hasattr(col,"__iter__" ):
            assert len(col) == len(ind_2_domains)
            c=col[i] #must match length
        else:
            c=colors[i] if col==None else col
        coords=zip(*pts)

        ax.plot(coords[0],coords[1],color=c,lw=2,alpha=.75)

    return ax



def create_similarity_heatmap_single_layer(partitions,index_2_domain):
    '''
    todo

    :param partitions:
    :param index_2_domain:
    :return:
    '''
    #TODO
    return