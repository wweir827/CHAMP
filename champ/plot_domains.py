import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from .champ_functions import create_halfspaces_from_array

def plot_line_coefficients(coef_array,ax=None,colors=None):
    '''
    Plot an array of coefficients (lines) in 2D plane.  Each line is drawn from \
     y-intercept to x-intercept.

    :param coef_array: :math:`N\\times 2` array of coefficients representing lines.
    :param ax: optional matplotlib ax to draw the figure on.
    :param cols: optional list of colors (or single color) to draw lines
    :return: matplotlib ax on which the plot is draw

    '''
    halfspaces=create_halfspaces_from_array(coef_array)
    ax=plot_line_halfspaces(halfspaces,ax=ax,colors=colors)
    return ax

def plot_line_halfspaces(halfspaces, ax=None, colors=None):
    '''
    Plot a list of halfspaces (lines) in 2D plane.  Each line is drawn from y-intercept to x-intercept.

    :param halfspaces: list of halfspaces
    :param ax: optional matplotlib ax to draw the figure on
    :param cols: optional list of colors (or single color) to draw lines
    :return: matplotlib ax on which the plot is draw

    '''

    if ax == None:
        f = plt.figure()
        ax = f.add_subplot(111)

    if colors==None:
        cnorm=mcolors.Normalize(vmin=0,vmax=len(halfspaces))
        cmap=cm.get_cmap("Set1")
        pal=map(lambda(i): cmap(cnorm(i)),range(len(halfspaces)))

    for i,hs in enumerate(halfspaces):
        if hasattr(colors, "__iter__"):
            c = colors[i%len(colors)]  # must match length
        else:
            c = pal[i] if colors == None else colors

        A=-1.0*hs.offset/hs.normal[0]
        B=-1.0*hs.offset/hs.normal[1]
        ax.plot( [0,A],[B,0],color=c )

    return ax


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
        xcrds=[x[0] for x in  pts ]
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

    :param ind_2_domains: dictionary mapping partition index to domain of dominance
    :type ind_2_domains: { ind: [ np.array(gam_0x, gam_0y), np.array(gam_1x, gam_1y)  ] ,...}
    :param ax: Matplotlib Axes object to draw the graph on
    :param col: Either a single color or list of colors with same length as number of domains
    :return :  ax  Reference to the ax on which plot is drawn.

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

    ax.set_ylabel("Q,Modularity")
    ax.set_xlabel("resolution")

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