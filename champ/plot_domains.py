import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from sklearn.metrics import adjusted_mutual_info_score,normalized_mutual_info_score
from .champ_functions import create_halfspaces_from_array
from future.utils import iteritems,iterkeys
from future.utils import lmap
import seaborn as sbn

def plot_line_coefficients(coef_array,ax=None,colors=None):
    '''
    Plot an array of coefficients (lines) in 2D plane.  Each line is drawn from \
     y-intercept to x-intercept.

    :param coef_array: :math:`N\\times 2` array of coefficients representing lines.
    :type coef_array: np.array
    :param ax: optional matplotlib ax to draw the figure on.
    :type ax: matplotlib.Axes
    :param colors: optional list of colors (or single color) to draw lines
    :type colors: [list,string]
    :return: matplotlib ax on which the plot is draw

    '''
    halfspaces=create_halfspaces_from_array(coef_array)
    ax=plot_line_halfspaces(halfspaces,ax=ax,colors=colors)
    return ax

def plot_line_halfspaces(halfspaces, ax=None, colors=None, labels=None):
    '''
    Plot a list of halfspaces (lines) in 2D plane.  Each line is drawn from y-intercept to x-intercept.

    :param halfspaces: list of halfspaces
    :param ax: optional matplotlib ax to draw the figure on
    :param cols: optional list of colors (or single color) to draw lines
    :param labels: Include labels of axes.  Either True or tuple of strings
    :type labels:bool,(xlabel,ylabel)
    :return: matplotlib ax on which the plot is draw

    '''

    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)

    if colors is None:
        cnorm = mcolors.Normalize(vmin=0, vmax=len(halfspaces))
        cmap = cm.get_cmap("Set1")
        pal = lmap(lambda i: cmap(cnorm(i)), range(len(halfspaces)))

    normals, offsets = np.split(halfspaces, [-1], axis=1)

    for i, (normal, offset) in enumerate(zip(normals, offsets)):
        if hasattr(colors, "__iter__"):
            c = colors[i % len(colors)]  # must match length
        else:
            c = pal[i] if colors is None else colors

        A = -1.0 * offset / normal[0]
        B = -1.0 * offset / normal[1]
        ax.plot([0, A], [B, 0], color=c)

    if labels is not None:
        if labels is True:
            ax.set_ylabel("Modularity")
            ax.set_xlabel("resolution")
        else:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])

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
    if widths==None: #we use different line widths for distinguishablity
        widths = np.random.sample(len(ind_2_domains))*3 + 1
    if col==None:
        cnorm=mcolors.Normalize(vmin=0,vmax=len(ind_2_domains))
        cmap=cm.get_cmap("Set1")
        colors=lmap(lambda i: cmap(cnorm(i)),range(len(ind_2_domains)))
    i=0

    for i,ind_pts in enumerate(iteritems(ind_2_domains)):
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

def plot_single_layer_modularity_domains(ind_2_domains, ax=None, colors=None, labels=None):
    '''
    Plot the piece-wise linear curve for CHAMP of single layer partitions

    :param ind_2_domains: dictionary mapping partition index to domain of dominance
    :type ind_2_domains: { ind: [ np.array(gam_0x, gam_0y), np.array(gam_1x, gam_1y)  ] ,...}
    :param ax: Matplotlib Axes object to draw the graph on
    :param colors: Either a single color or list of colors with same length as number of domains
    :return: ax  Reference to the ax on which plot is drawn.

    '''
    if ax==None:
        f=plt.figure()
        ax=f.add_subplot(111)

    if colors==None:
        cnorm=mcolors.Normalize(vmin=0,vmax=len(ind_2_domains))
        cmap=cm.get_cmap("Set1")
        pal=lmap(lambda i : cmap(cnorm(i)),range(len(ind_2_domains)))
    i=0

    for i,ind_pts in enumerate(iteritems(ind_2_domains)):
        ind, pts = ind_pts
        if hasattr(colors, "__iter__"):
            assert len(colors) == len(ind_2_domains)
            c=colors[i] #must match length
        else:
            c=pal[i] if colors == None else colors
        coords=lmap(list,zip(*pts))


        ax.scatter(coords[0],coords[1],color=c,marker='x')
        ax.plot(coords[0],coords[1],color=c,lw=2,alpha=.75)

    if labels is not None:
        if labels is True:
            ax.set_ylabel("modularity")
            ax.set_xlabel("resolution")
        else:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])

    return ax



def plot_similarity_heatmap_single_layer(partitions, index_2_domain, partitions_other=None,index_2_dom_other=None,
                                         sim_mat=None,ax=None, cmap=None, title=None):
    '''

    :param partitions:
    :type partitions:
    :param index_2_domain:
    :type index_2_domain:
    :param sim_mat:
    :type sim_mat:
    :param ax: Axes to draw the figure on.  New figure created if not supplied.
    :type ax: Matplotlib.Axes
    :param cmap: Color mapping.  Default is plasma
    :type cmap: Matplotlib.colors.Colormap
    :param title: True if add generic title, or string of title to add
    :type title: boolean or string
    :return: axis drawn on , computed similarity matrix
    :rtype: matplolib.Axes,np.array

    '''

    if cmap is None:
        cmap = 'PuBu'

    if ax is None:
        f=plt.figure()
        ax=f.add_subplot(111)

    ind_vals=list(zip(index_2_domain.keys(),[val[0][0] for val in index_2_domain.values()]))
    ind_vals.sort(key=lambda x:x[1])
    # Take the x coordinate of first point in each domain
    gamma_transitions = [val for ind, val in ind_vals]
    gamma_transitions.append(index_2_domain[ind_vals[-1][0]][-1][0])  # Append last point of last partition

    #if there isn't a partition set to compare to the default is to compare to self.
    if not( index_2_dom_other is None or partitions_other is None):
        ind_vals2 = zip(index_2_dom_other.keys(), [val[0][0] for val in index_2_dom_other.values()])
        ind_vals2.sort(key=lambda x: x[1])
        gamma_transitions2 = [val for ind, val in ind_vals2]
        gamma_transitions2.append(index_2_dom_other[ind_vals2[-1][0]][-1][0])
    else:
        ind_vals2 = ind_vals
        gamma_transitions2=gamma_transitions
        partitions_other=partitions


    # G2S, G1S = np.meshgrid(gamma_transitions2, gamma_transitions)
    G1S, G2S = np.meshgrid(gamma_transitions2, gamma_transitions)
    if sim_mat is None:
        AMI_mat = np.zeros((len(ind_vals), len(ind_vals2)))
        for i  in range(len(ind_vals)):
            for j in range(len(ind_vals2)):
                partition1=partitions[ind_vals[i][0]]
                partition2=partitions_other[ind_vals2[j][0]]

                AMI_mat[i][j] = adjusted_mutual_info_score(partition1,
                                                            partition2)
                # AMI_mat[j][i] = AMI_mat[i][j] #symmetric
    else:
        AMI_mat = sim_mat


    # pal = sbn.cubehelix_palette(start=2, rot=.3, dark=0, light=.95, reverse=True, as_cmap=True)


    pmap = ax.pcolor(G1S, G2S, AMI_mat, cmap=cmap)
    ax.set_xlabel("resolution")
    ax.set_ylabel("resolution")

    for gm in gamma_transitions:
        ax.axhline(gm, color='k', linestyle='dashed')
    for gm in gamma_transitions2:
        ax.axvline(gm, color='k', linestyle='dashed')

    ax.set_ylim(gamma_transitions[0], max(gamma_transitions))
    ax.set_xlim(gamma_transitions2[0], max(gamma_transitions2))
    if not title is None:
        if title is True:
            ax.set_title('Adjusted Mutual Information between Partitions')
        else:
            ax.set_title(title)
    plt.colorbar(pmap, ax=ax)
    return ax, AMI_mat

def _get_partition_matrix(partition, layer_vec):
    # assumes partiton in same ordering for each layer
    vals = np.unique(layer_vec)
    nodeperlayer = len(layer_vec) // len(vals) #integerdevision
    com_matrix = np.zeros((nodeperlayer, len(vals)))
    for i, val in enumerate(vals):
        cind = np.where(layer_vec == val)[0]
        ccoms = partition[cind]
        com_matrix[:, i] = ccoms
    return com_matrix

def plot_multiplex_community(partition, layer_vec, ax=None, cmap=None):

    """ This is for visualizing community of mutliplex """
    part_mat = _get_partition_matrix(partition, layer_vec)
    layers=np.unique(layer_vec)
    if ax is None:
        ax = plt.axes()

    if cmap is None:
        cmap = sbn.cubehelix_palette(as_cmap=True)

    ax.grid('off')
    ax.pcolormesh(part_mat, cmap=cmap)

    ax.set_xticks(range(0, len(layers)))
    ax.set_xticklabels(layers)
    return ax



