import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patch
from matplotlib.path import Path


def plot_2d_domains(ind_2_domains, ax=None, col=None, close=False, widths=None, label=False):
    if ax==None:
        f=plt.figure()
        ax=f.add_subplot(111)
    if widths==None:
        widths = np.random.sample(len(plane_dict))*3 + 1
    if col==None:
        cnorm=mcolors.Normalize(vmin=0,vmax=len(plane_dict.keys()))
        cmap=cm.get_cmap("Set1")
        colors=map(lambda(i): cmap(cnorm(i)),range(len(plane_dict)))
    i=0
    for i,pts in ind_2_domains.items():
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
        i+=1
    return ax