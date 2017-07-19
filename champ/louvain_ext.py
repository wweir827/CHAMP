import gzip
import sys
import tempfile
from contextlib import contextmanager
from multiprocessing import Pool,cpu_count
from time import time

from .champ_functions import create_halfspaces_from_array
from .champ_functions import get_intersection

import igraph as ig
import louvain
import numpy as np

try:
    import cpickle as pickle
except ImportError:
    import pickle as pickle

@contextmanager
def terminating(obj):
    '''
    Context manager to handle appropriate shutdown of processes
    :param obj: obj to open
    :return:
    '''
    try:
        yield obj
    finally:
        obj.terminate()

'''
Extension of Traag's implementation of louvain in python to use multiprocessing \
and allow for randomization
'''

class PartitionEnsemble():

    '''Group of partitions of a graph stored in membership vector format



    The attribute for each partition is stored in an array and can be indexed

    :cvar graph: The graph associated with this PartitionEnsemble.  Each ensemble \
    can only have a single graph and the nodes on the graph must be orded the same as \
    each of the membership vectors.
    :type graph: igraph.Graph

    :cvar partitions: List of membership vectors for each partition

    :cvar int_edges:  Number of edges internal to the communities
    :type int_edges:  list

    :cvar exp_edges:  Number of expected edges (based on configuration model)
    :type exp_edges:  list

    :cvar resoltions:  If partitions were idenitfied with Louvain, what resolution \
    were they identified at (otherwise None)
    :type resolutions: list

    :cvar orig_mods:  Modularity of partition at the resolution it was identified at \
    if Louvain was used (otherwise None).
    :type orig_mods: list

    :cvar numparts: number of partitions
    :type numparts: int
    :cvar ind2doms: Maps index of dominant partitions to boundary points of their dominant \
    domains
    :type ind2doms: dict

    '''


    def __init__(self,graph=None,listofparts=None,name='unnamed_graph',maxpt=None):
        self.partitions = []
        self.int_edges = []
        self.exp_edges = []
        self.resolutions = []
        self.orig_mods = []
        self.numparts=0
        self.graph=graph
        if listofparts!=None:
            self.add_partitions(listofparts,maxpt=maxpt)
        self.name=name


    def get_adjacency(self):
        '''
        Calc adjacency representation if it exists

        :return: self.adjacency

        '''
        if self.adjacency==None:
            if 'weight' in self.graph.edge_attributes():
                self.adjacency=self.graph.get_adjacency(type="GET_ADJACENCY_BOTH",
                                                    attribute='weight')
            else:
                self.adjacency = self.graph.get_adjacency(type="GET_ADJACENCY_BOTH")
        return self.adjacency

    def calc_internal_edges(self,memvec):
        '''
        Uses igraph Vertex Clustering representation to calculate internal edges.  see \
        :meth:`louvain_ext.get_expected_edges`

        :param memvec: membership vector for which to calculate the internal edges.
        :type memvec: list

        :return:

        '''
        # if "weight" in self.graph.edge_attributes():
        #     adj=self.graph.get_adjacency(attribute='weight')
        partobj = ig.VertexClustering(graph=self.graph, membership=memvec)
        weight = "weight" if "weight" in self.graph.edge_attributes() else None
        return get_sum_internal_edges()

    def calc_expected_edges(self, memvec):
        '''
        Uses igraph Vertex Clustering representation to calculate expected edges.  see \
        :meth:`louvain_ext.get_expected_edges`

        :param memvec: membership vector for which to calculate the expected edges
        :type memvec: list
        :return: expected edges under null
        :rtype: float

        '''
        # adj = self.graph.as_adjacency()
        # m=np.sum(adj)
        # exp_adj = np.outer(self.graph.degree())

        #create temporary VC object
        partobj=ig.VertexClustering(graph=self.graph,membership=memvec)
        weight = "weight" if "weight" in self.graph.edge_attributes() else None
        return get_expected_edges(partobj,weight)


    def __getitem__(self, item):
        '''
        List of paritions in the PartitionEnsemble object can be indexed directly

        :param item: index of partition for direct access
        :type item: int
        :return: self.partitions[item]
        :rtype:  membership vector of community for partition
        '''
        return self.partitions[item]

    def _check_lengths(self):
        '''
        check all state variables for equal length

        :return: boolean indicating states varaible lengths are equal

        '''
        if len(self.partitions)==len(self.int_edges) and \
            len(self.partitions)==len(self.resolutions) and \
            len(self.partitions)==len(self.exp_edges):
            return True
        else:
            return False

    def add_partitions(self,partitions,maxpt=None):
        '''
        Add additional partitions to the PartitionEnsemble object

        :param partitions: list of partitions to add to the PartitionEnsemble
        :type partitions: dict,list

        '''
        #wrap in list.
        if not hasattr(partitions,'__iter__'):
            partitions=[partitions]

        for part in partitions:
            #This must be present
            self.partitions.append(part['partition'])

            if 'resolution' in part:
                self.resolutions.append(part['resolution'])
            else:
                self.resolutions.append(None)

            if 'int_edges' in part:
                self.int_edges.append(part['int_edges'])
            else:
                cint_edges=self.calc_internal_edges(part['partition'])
                self.int_edges.append(cint_edges)

            if 'exp_edges' in part:
                self.exp_edges.append(part['exp_edges'])
            else:
                cint_edges=self.calc_expected_edges(part['partition'])
                self.exp_edges.append(cint_edges)

            if "orig_mod" in part:
                self.orig_mods.append(part['orig_mod'])
            elif not self.resolutions[-1] is None:
                #calculated original modularity from orig resolution
                self.orig_mods.append(self.int_edges[-1]-self.resolutions[-1]*self.exp_edges)
            else:
                self.orig_mods.append(None)


            assert self._check_lengths()
            self.numparts=len(self.partitions)
            #update the pruned set
            self.apply_CHAMP(maxpt=maxpt)



    def get_partition_dictionary(self, ind=None):
        '''
        Get dictionary representation of partitions with the following keys:

            'partition','resolution','orig_mod','int_edges','exp_edges'

        :param ind: optional indices of partitions to return.  if not supplied all partitions will be returned.
        :type ind: int, list
        :return: list of dictionaries

        '''

        if ind is not None:
            if not hasattr(ind,"__iter__"):
                ind=[ind]
        else: #return all of the partitions
            ind=range(len(self.partitions))

        outdicts=[]
        for i in ind:
            cdict={"partition":self.partitions[i],
                   "int_edges":self.int_edges[i],
                   "exp_edges":self.exp_edges[i],
                   "resolution":self.resolutions[i],
                   "orig_mod":self.orig_mods[i]}
            outdicts.append(cdict)

        return outdicts

    def merge_ensemble(self,otherEnsemble):
        '''
        Combine to PartitionEnsembles.  Checks for concordance in the number of vertices. \
        Assumes that internal ordering on the graph nodes for each is the same.

        :param otherEnsemble: otherEnsemble to merge
        :return: new PartitionEnsemble with merged set of partitions

        '''

        if not self.graph.vcount()==otherEnsemble.graph.vcount():
            raise ValueError("PartitionEnsemble graph vertex counts do not match")

        bothpartitions=self.get_partition_dictionary().extend(otherEnsemble.get_partition_dictionary())


        return PartitionEnsemble(self.graph,listofparts=bothpartitions)

    def get_coefficient_array(self):
        '''
        Create array of coefficents for each partition

        :return: np.array with coefficents for each of the partions

        '''

        outlist=[]
        for i in range(len(self.partitions)):
            outlist.append([
                self.int_edges[i],
                self.exp_edges[i]
            ])

        return np.array(outlist)

    def apply_CHAMP(self,maxpt=None):
        '''
        Apply CHAMP to the partition ensemble.

        :param maxpt: maximum domain threshhold for included partition.  I.e \
        partitions with a domain greater than maxpt will not be included in pruned \
        set
        :type maxpt: int

        '''
        self.ind2doms=get_intersection(self.get_coefficient_array(),max_pt=maxpt)

    def get_CHAMP_indices(self):

        '''
        Get the indices of the partitions that form the pruned set after application of \
        CHAMP

        :return: list of indices of partitions that are included in the prune set \
        sorted by their domains of dominance
        :rtype: list

        '''

        inds=zip(self.ind2doms.keys(),[val[0][0] for val in self.ind2doms.values()])
        #asscending sort by last value of domain
        inds.sort(key=lambda x: x[1])

        #retreive index
        return [ind[0] for ind in inds]

    def get_CHAMP_partitions(self):

        '''Return the subset of partitions that form the outer envelop.
        :return: List of partitions in membership vector form of the paritions
        :rtype: list

        '''
        inds=self.get_CHAMP_indices()
        return [self.partitions[i] for i in inds]

    def save(self,filename=None):
        '''
        Use pickle to dump representation to compressed file

        :param filename:

        '''
        if filename is None:
            filename="%s_PartEnsemble_%d.gz" %(self.name,self.numparts)

        with gzip.open(filename,'wb') as fh:
            pickle.dump(self,fh)


    @staticmethod
    def open(filename):
        '''
        Loads pickled PartitionEnsemble from file.

        :param file:  filename of pickled PartitionEnsemble Object
        :return: writes over current instance and returns the reference

        '''
        with gzip.open(filename,'rb') as fh:
            opened=pickle.load(fh)

        openedparts=opened.get_partition_dictionary()

        #construct and return
        return PartitionEnsemble(opened.graph,listofparts=openedparts)




##### STATIC METHODS ######

def get_sum_internal_edges(partobj,weight=None):
    '''
       Get the count(strength) of edges that are internal to community:

       :math:`\\hat{A}=\\sum_{ij}{A_{ij}\\delta(c_i,c_j)}`

       :param partobj:
       :type partobj: igraph.VertexClustering
       :param weight: True uses 'weight' attribute of edges
       :return: float
       '''
    sumA=0
    for subg in partobj.subgraphs():
        if weight!=None:
            sumA+= np.sum(subg.es[weight])
        else:
            sumA+= subg.ecount()
    return 2.0*sumA

def get_expected_edges(partobj,weight=None):
    '''
    Get the expected internal edges under configuration models

    :math:`\\hat{P}=\\sum_{ij}{\\frac{k_ik_j}{2m}\\delta(c_i,c_j)}`

    :param partobj:
    :type partobj: igraph.VertexClustering
    :param weight: True uses 'weight' attribute of edges
    :return: float
    '''

    if weight==None:
        m = partobj.graph.ecount()
    else:
        m=np.sum(partobj.graph.es['weight'])
    kk=0
    #Hashing this upfront is alot faster (factor of 10).
    if weight==None:
        strengths=dict(zip(partobj.graph.vs['id'],partobj.graph.degree(partobj.graph.vs)))
    else:
        strengths=dict(zip(partobj.graph.vs['id'],partobj.graph.strength(partobj.graph.vs,weights="weight")))
    for subg in partobj.subgraphs():
        # since node ordering on subgraph doesn't match main graph, get vert id's in original graph
        # verts=map(lambda x: int(re.search("(?<=n)\d+", x['id']).group()),subg.vs) #you have to get full weight from original graph
        # svec=partobj.graph.strength(verts,weights='weight') #i think is what is slow

        svec=np.array(map(lambda(x):strengths[x],subg.vs['id']))
        # svec=subg.strength(subg.vs,weights='weight')
        kk+=np.sum(np.outer(svec, svec))

    return kk/(2.0*m)

def rev_perm(perm):
    '''
    Calculate the reverse of a permuation vector

    :param perm: permutation vector
    :type perm: list
    :return: reverse of permutation
    '''
    rperm=list(np.zeros(len(perm)))
    for i,v in enumerate(perm):
        rperm[v]=i
    return rperm

def get_orig_ordered_mem_vec(rev_order, membership):
    '''
    Rearrange community membership vector according to permutation

    Used to realign community vector output after node permutation.

    :param rev_order: new indices of each nodes
    :param membership: community membership vector to be rearranged
    :return: rearranged membership vector.
    '''
    new_member=[-1 for i in xrange(len(rev_order))]

    for i,val in enumerate(rev_order):
        new_member[val]=membership[i]
    assert(-1 not in new_member) #Something didn't get switched

    return new_member

def run_louvain(gfile,gamma,nruns,weight=None,node_subset=None,attribute=None,output_dictionary=False):
    '''
    Call the louvain method for a given graph file.

    This takes as input a graph file (instead of the graph object) to avoid duplicating
    references in the context of parallelization.  To allow for flexibility, it allows for
    subsetting of the nodes each time.

    :param gfile: igraph file.  Must be GraphMlz (todo: other extensions)
    :param node_subset: Subeset of nodes to keep (either the indices or list of attributes)
    :param gamma: resolution parameter to run louvain
    :param nruns: number of runs to conduct
    :param weight: optional name of weight attribute for the edges if network is weighted.
    :param output_dictionary: Boolean - output a dictionary representation without attached graph.
    :return: list of partition objects

    '''

    np.random.seed() #reset seed for each process

    #Load the graph from the file
    g = ig.Graph.Read_GraphMLz(gfile)
    #have to have a node identifier to handle permutations.


    #Found it easier to load graph from file each time than pass graph object among process
    #This means you do have to filter out shared nodes and realign graphs.
    # Can avoid for g1 by passing None

    #
    if node_subset!=None:
        # subset is index of vertices to keep
        if attribute==None:
            gdel=node_subset
        # check to keep nodes with given attribute
        else:
            gdel=[ i for i,val in enumerate(g.vs[attribute]) if val not in node_subset]

        #delete from graph
        g.delete_vertices(gdel)


    outparts=[]
    for i in xrange(nruns):
        rand_perm = list(np.random.permutation(g.vcount()))
        rperm = rev_perm(rand_perm)
        gr=g.permute_vertices(rand_perm) #This is just a labelling switch.  internal properties maintined.
        rp = louvain.find_partition(gr, method='RBConfiguration',weight=weight,  resolution_parameter=gamma)

        #store the coefficients in return object.
        A=get_sum_internal_edges(rp,weight)
        P=get_expected_edges(rp,weight)

        outparts.append({'partition': get_orig_ordered_mem_vec(rperm, rp.membership),
                         'resolution':gamma,
                         'orig_mod': rp.quality,
                         'int_edges':A,
                         'exp_edges':P})

    if not output_dictionary:
        return PartitionEnsemble(graph=g,listofparts=outparts)
    else:
        return outparts
    return part_ensemble





def _run_louvain_parallel(gfile_gamma_nruns_weight_subset_attribute_progress):
    '''
    Parallel wrapper with single argument input for calling :meth:`louvain_ext.run_louvain`

    :param gfile_att_2_id_dict_shared_gamma_runs_weight: tuple or list of arguments to supply
    :returns: PartitionEnsemble of graph stored in gfile
    '''
    #unpack
    gfile,gamma,nruns,weight,node_subset,attribute,progress=gfile_gamma_nruns_weight_subset_attribute_progress
    t=time()
    outparts=run_louvain(gfile,gamma,nruns=nruns,weight=weight,node_subset=node_subset,attribute=attribute,output_dictionary=True)

    if progress is not None:
        if progress%100==0:
            print "Run %d at gamma = %.3f.  Return time: %.4f" %(progress,gamma,time()-t)

    return outparts
def parallel_louvain(graph,start=0,fin=1,numruns=200,maxpt=None,
                     numprocesses=None,attribute=None,weight=None,node_subset=None,progress=False):
    '''
    Generates arguments for parallel function call of louvain on graph

    :param graph: igraph object to run Louvain on
    :param start: beginning of range of resolution parameter :math:`\\gamma` . Default is 0.
    :param fin: end of range of resolution parameter :math:`\\gamma`.  Default is 1.
    :param numruns: number of intervals to divide resolution parameter, :math:`\\gamma` range into
    :param maxpt: Cutoff off resolution for domains when applying CHAMP. Default is None
    :type maxpt: int
    :param numprocesses: the number of processes to spawn.  Default is number of CPUs.
    :param weight: If True will use 'weight' attribute of edges in runnning Louvain and calculating modularity.
    :param node_subset:  Optionally list of indices or attributes of nodes to keep while partitioning
    :param attribute: Which attribute to filter on if node_subset is supplied.  If None, node subset is assumed \
     to be node indices.
    :param progress:  Print progress in parallel execution
    :return: PartitionEnsemble of all partitions identified.

    '''
    parallel_args=[]
    if numprocesses is None:
        numprocesses=cpu_count()


    tempf=tempfile.NamedTemporaryFile('wb')
    graphfile=tempf.name
    #filter before calling parallel
    if node_subset != None:
        # subset is index of vertices to keep
        if attribute == None:
            gdel = node_subset
        # check to keep nodes with given attribute
        else:
            gdel = [i for i, val in enumerate(graph.vs[attribute]) if val not in node_subset]

        # delete from graph
        graph.delete_vertices(gdel)

    graph.write_graphmlz(graphfile)
    for i in xrange(numruns):
        prognum = None if not progress else i
        curg = start + ((fin - start) / (1.0 * numruns)) * i
        parallel_args.append((graphfile ,curg,1, weight,None,None,prognum))


    #use a context manager so pools properly shut down

    with terminating(Pool(processes=numprocesses)) as pool:
        parts_list_of_list=pool.map(_run_louvain_parallel, parallel_args )


    all_part_dicts=[pt for partrun in parts_list_of_list for pt in partrun]

    outensemble=PartitionEnsemble(graph,listofparts=all_part_dicts,maxpt=maxpt)
    return outensemble


def main():
    return

if __name__=="__main__":
     sys.exit(main())