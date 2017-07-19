import multiprocessing
from collections import Hashable

import numpy as np
from numpy.random import uniform
from pyhull import halfspace as hs


def create_coefarray_from_partitions(partition_array, A_mat, P_mat, C_mat=None,nprocesses=0):
    '''
   :param partition_array: Each row is one of M partitions of the network with N nodes.  Community labels must be hashable.
   :param A_mat: Interlayer (single layer) adjacency matrix
   :param P_mat: Matrix representing null model of connectivity (i.e configuration model - :math:`\\frac{k_ik_j}{2m}`
   :param C_mat: Optional matrix representing interlayer connectivity
   :param nprocesses: Optional number of processes to use (0 or 1 for single core)
   :type nprocesses: int
   :return: size :math:`M\\times\\text{Dim}` array of coefficients for each partition. Dim can be 2 (single layer) \
   or 3 (multilayer)

    '''
    outarray = []

    if nprocesses==0 or nprocesses==1:


        for partition in partition_array:
            curarray=[]
            curarray.append(calculate_coefficient(partition,A_mat))
            curarray.append(calculate_coefficient(partition,P_mat))
            curarray.append(calculate_coefficient(partition,C_mat))
            outarray.append(curarray)

    else:
        pool=multiprocessing.Pool(nprocesses=nprocesses)
        parallel_args=[]
        for partition in partition_array:
            parallel_args.append((partition, A_mat))
            parallel_args.append((partition, P_mat))
            parallel_args.append((partition, C_mat))
        #map preserves order
        parallel_res=pool.map(_calculate_coefficient_parallel,parallel_args)
        outarray=np.array(parallel_res).reshape((3,len(parallel_res)/3))
    return np.array(outarray)

def create_halfspaces_from_array(coef_array):
    '''
    :param coef_array: list of coefficients for each partition to be considered.  Should be an array
    .. math:
        :nowrap:

        \begin{array}
        1 & 1 & 1 \\
        2 & 2 & 2
        \end{array}
        ]

    Where each row represents the coefficients for a particular partition.
    If Single Layer network, omit C_i's.
    :return: list of halfspaces with 4 boundary halfspaces appended to the end.

    '''
    singlelayer=False
    if coef_array.shape[1]==2:
        singlelayer=True


    halfspaces=[]
    for i in np.arange(coef_array.shape[0]):
        cconst = coef_array[i, 0]
        cgamma = coef_array[i, 1]
        if not singlelayer:
            comega = coef_array[i, 2]

        if singlelayer:
            nv=np.array([cgamma, 1.0])
            pt = np.array([0 , cconst])

        else:
            nv=np.array([cgamma, -1 * comega, 1.0])
            pt = np.array([0, 0, cconst])

        nv = nv/np.linalg.norm(nv)
        off = np.dot(nv, pt)

        halfspace = hs.Halfspace(-1.0 * nv, off)
        halfspaces.append(halfspace)



    return halfspaces

def sort_points(points):
    '''

    :param points:
    :return:
    '''
    if len(points[0])>2:
        cent = (sum([p[0] for p in points]) / len(points), sum([p[1] for p in points]) / len(points))
        points.sort(key=lambda (x): np.arctan2(x[1] - cent[1], x[0] - cent[0]))
    else:
       points.sort(key=lambda (x): x[0]) #just sort along x-axis


    return points

def get_interior_point(hs_list):
    '''
    Find interior point to calculate intersections
    :param hs_list: list of halfspaces
    :return: interior point of intersections at (0+.001,0+.001,max(A_ij)) the
    maximum modularity value of any of the partitions at 0,0 needed to calculate
    interior intersection.

    '''

    z_vals=[ -1.0*hs.offset/hs.normal[-1] for hs in hs_list if
             np.abs(hs.normal[-1]-0)>np.power(10.0,-15)]

    #take a small step into interior from 1st plane.
    dim = hs_list[0].normal.shape[0]
    intpt=np.array([0 for _ in range(dim-1)]+[np.max(z_vals)])
    internal_step=np.array([.000001 for _ in range(dim)])
    return intpt+internal_step




def calculate_coefficient(com_vec,adj_matrix):
    '''
    For a given connection matrix and set of community labels, calculate the coeffcient
    for plane/line associated with that connectivity matrix

    :param com_vec: list or vector with community membership for each element of network
    ordered the same as the rows/col of adj_matrix
    :param adj_matrix: adjacency matrix for connections to calculate coefficients for
    (i.e. A_ij, P_ij, C_ij, etc..) ordered the same as com_vec
    :return:

    '''

    com_inddict = {}

    allcoms = sorted(list(set(com_vec)))
    assert com_vec.shape[0] == adj_matrix.shape[0]
    sumA = 0

    # store indices for each community together in dict
    for i, val in enumerate(com_vec):
        try:
            com_inddict[val] = com_inddict.get(val, []) + [i]
        except TypeError:
            raise TypeError ("Community labels must be hashable- isinstance(%s,Hashable): " %(str(val)),\
                             isinstance(val,Hashable))

    # convert indices to np_array
    for k, val in com_inddict.items():
        com_inddict[k] = np.array(val)

    for com in allcoms:
        cind = com_inddict[com]
        if cind.shape[0] == 1:  # throws type error if try to index with scalar
            sumA += np.sum(adj_matrix[cind, cind])
        else:
            sumA += np.sum(adj_matrix[np.ix_(cind, cind)])

    return sumA

def _calculate_coefficient_parallel(comvec_mat):
    '''
    wrapper function for calc coefficient with single parameter for use with the \
    multiprocessing map call

    :param comvec_mat: (community vector, adj_matrix
    :return: calculate_coefficient to get coeficient
    '''
    com_vec,adj_matrix=comvec_mat

    return calculate_coefficient(com_vec,adj_matrix)

def comp_points(pt1,pt2):
    '''
    check for equality within certain tolerance
    :param pt1:
    :param pt2:
    :return:

    '''
    for i in range(len(pt1)):
        if np.abs(pt1[i]-pt2[i])>np.power(10.0,-15):
            return False

    return True


def get_intersection(coef_array, max_pt=None):
    '''
    Calculate the intersection of the halfspaces (planes) that form the convex hull

   :param coef_array: NxM array of M coefficients across each row representing N partitions
   :type coef_array: array
   :param max_pt: Upper bound for the domains (in the xy plane). This will restrict the convex hull \
    to be within the specified range of gamma/omega (such as the range of parameters originally searched using Louvain).
   :type max_pt: (float,float)
   :return: dictionary mapping the index of the elements in the convex hull to the points defining the boundary
    of the domain
    '''


    halfspaces=create_halfspaces_from_array(coef_array)

    interior_pt=get_interior_point(halfspaces)
    singlelayer=False
    if halfspaces[0].normal.shape[0]==2:
        singlelayer=True


    # Create Boundary Halfspaces - These will always be included in the convex hull
    # and need to be removed before returning

    num2rm=0
    if not singlelayer:
        # origin boundaries
        halfspaces.extend([hs.Halfspace(normal=(0, -1.0, 0), offset=0), hs.Halfspace(normal=(-1.0, 0, 0), offset=0)])
        num2rm +=2
        if max_pt is not None:
            halfspaces.extend([hs.Halfspace(normal=(0, 1.0, 0), offset=-1.0 * max_pt[0]),
                           hs.Halfspace(normal=(1.0, 0, 0), offset=-1 * max_pt[1])])
            num2rm +=2

    else:
        halfspaces.append(hs.Halfspace(normal=(-1,0), offset=0)) # y-axis
        halfspaces.append(hs.Halfspace(normal=(0,-1), offset=0)) # x-axis

        num2rm +=2
        if max_pt is not None:
            halfspaces.append(hs.Halfspace(normal=(1.0, 0), offset=-1 * max_pt))
            num2rm+=1


    hs_inter = hs.HalfspaceIntersection(halfspaces, interior_pt)  # Find boundary intersection of half spaces
    ind_2_domain = {}

    non_inf_vert = np.array([v for v in hs_inter.vertices if v[0] != np.inf])
    mx = np.max(non_inf_vert,axis=0)

    #At this point we inlude max boundary planes and recalculate the intersection
    #to correct inf points
    if max_pt is None:
        if not singlelayer:
            halfspaces.extend([hs.Halfspace(normal=(0, 1.0, 0), offset=-1.0 * (mx[0])),
                               hs.Halfspace(normal=(1.0, 0, 0), offset=-1.0* (mx[0]) )])
            num2rm += 2

    if not singlelayer:
        hs_inter = hs.HalfspaceIntersection(halfspaces, interior_pt)  # Find boundary intersection of half spaces

    # assert not any([ coord==np.inf  for v in hs_inter.vertices for coord in v ])


    rep_verts = [v if v[0] != np.inf else mx for v in hs_inter.vertices]
    # rep_verts=hs_inter.vertices

    for i, vlist in enumerate(hs_inter.facets_by_halfspace):
        #Empty domains
        if len(vlist) == 0:
            continue

        # these are the boundary planes appended on end
        if not i < len(hs_inter.facets_by_halfspace) - num2rm :
            continue

        pts=sort_points([rep_verts[v] for v in vlist])
        pt2rm=[]
        for j in range(len(pts)-1):
            if comp_points(pts[j],pts[j+1]):
                pt2rm.append(j)
        pt2rm.reverse()
        for j in pt2rm:
            pts.pop(j)
        if len(pts)>=len(rep_verts[0]): #must be at least 2 pts in 2D , 3 pt in 3D, etc.
            ind_2_domain[i]=pts

        #use non-inf vertices to return
    return ind_2_domain


def _random_plane():

    normal = np.array([uniform(-0.5, 0.5), uniform(-0.5, 0.5), -1])
    normal /= np.linalg.norm(normal)
    min_offset = -min(0,
                      normal[0],
                      normal[1],
                      normal[0] + normal[1])
    max_offset = -max(normal[2],
                      normal[2] + normal[0],
                      normal[2] + normal[1],
                      normal[2] + normal[0] + normal[1])
    # the 0.25 and 0.75 factors here just force more intersections
    offset = uniform(0.75 * min_offset + 0.25 * max_offset,
                     0.25 * min_offset + 0.75 * max_offset)

    #Return a coefficient representation instead
    return np.array([normal[0],normal[1],-1*offset/normal[2]])
    # return hs.Halfspace(normal, offset)

def _random_line():
    '''
    generate a random line in gamma,Q plane
    :return:
    '''
    # normal = np.array([uniform(.5, 2),-1])
    # normal /= np.linalg.norm(normal)

    #just sample slope and intercept directly
    slope=uniform(1/5.0,5)
    inter=uniform(0,2)

    # offset=-1.0*normal.dot(pt)

    # the 0.25 and 0.75 factors here just force more intersections
    # Return a coefficient representation instead
    return np.array([inter, slope])

def get_random_halfspaces(n=100,dim=3):
    '''Generate random halfspaces for testing
    :param n: number of halfspaces to return (default=100)
    '''
    test_hs=[]
    for _ in range(n):
        if dim==3:
            test_hs.append(_random_plane())
        elif dim==2:
            test_hs.append(_random_line())
        else:
            raise NotImplementedError("Only 2D or 3D Random Halfspaces implemented")
    return np.array(test_hs)
    # return test_hs