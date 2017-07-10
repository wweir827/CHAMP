import matplotlib.pyplot as plt
import numpy as np



def plot_2d_domains(plane_dict, ax=None, col=None, close=False, widths=None, label=False):
    if ax==None:
        f=plt.figure()
        ax=f.add_subplot(111)
    if widths==None:
        widths = np.random.sample(len(plane_dict))*3 + 1
    if col==None:
        cnorm=mc.Normalize(vmin=0,vmax=len(plane_dict.keys()))
        cmap=cm.get_cmap("Set1")
        colors=map(lambda(i): cmap(cnorm(i)),range(len(plane_dict)))
    i=0
    for plane,pts in plane_dict.items():
        if hasattr(col,"__iter__" ):
            assert len(col) == len(plane_dict)
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