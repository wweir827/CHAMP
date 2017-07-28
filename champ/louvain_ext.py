#Py 2/3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division # use // to specify int div.
from future.utils import iteritems,iterkeys
from future.utils import lmap

import gzip
import sys, os
import re
import tempfile
from contextlib import contextmanager
from multiprocessing import Pool,cpu_count
from time import time
from .champ_functions import get_intersection
from .champ_functions import create_coefarray_from_partitions
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import rc
import igraph as ig
import louvain
import numpy as np
import h5py


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
and allow for randomization.  Defines PartitionEnsemble a class for storage of \
partitions and coefficients as well as dominant domains.
'''

class PartitionEnsemble():

    '''Group of partitions of a graph stored in membership vector format



    The attribute for each partition is stored in an array and can be indexed

    :cvar graph: The graph associated with this PartitionEnsemble.  Each ensemble \
    can only have a single graph and the nodes on the graph must be orded the same as \
    each of the membership vectors.
    :type graph: igraph.Graph

    :cvar partitions:  of membership vectors for each partition.  If h5py is set this is a dummy \
    variable that allows access to the file, but never actually hold the array of parititons.
    :type partitions:  np.array
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
    :cvar ncoms: List with number of communities for each partition
    :type numcoms: list
    :cvar min_com_size: How many nodes must be in a community for it to count towards the number \
    of communities.  This eliminates very small or unstable communities.  Default is 5
    :type min_com_size: int
    :cvar unique_partition_indices: The indices of the paritions that represent unique coefficients.  This will be a \
    subset of all the partitions.
    :cvar hdf5_file: Current hdf5_file.  If not None, this serves as the default location for loading and writing \
    partitions to, as well as the default location for saving.
    :type hdf5_file: str
    :type unique_partition_indices: np.array
    :cvar twin_partitions:  We define twin partitions as those that have the same coefficients, but are actually \
    different partitions.  This and the unique_partition_indices are only calculated on demand which can take some time.
    :type twi_partitions: list of np.arrays
    '''


    def __init__(self,graph=None,listofparts=None,name='unnamed_graph',maxpt=None,min_com_size=5):

        self.hdf5_file=None
        self.int_edges = np.array([])
        self.exp_edges = np.array([])
        self.resolutions = np.array([])
        self.numcoms=np.array([])
        self.orig_mods = np.array([])
        self.numparts=0
        self.graph=graph
        self.min_com_size=min_com_size
        self.maxpt=maxpt
        #some private variable
        self._partitions=np.array([])
        self._uniq_coeff_indices=None
        self._uniq_partition_indices=None
        self._twin_partitions=None

        if listofparts!=None:
            self.add_partitions(listofparts,maxpt=self.maxpt)
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
        return get_sum_internal_edges(partobj=partobj,weight=weight)

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

    class _PartitionOnFile():

        def __init__(self,file=None):
            self.hdf5_file=file

        def __getitem__(self, item):
            with h5py.File(self.hdf5_file, 'r') as openfile:
                try:
                    return  openfile['_partitions'].__getitem__(item)
                except TypeError:
                    #h5py has some controls on what can be used as a slice object.
                    return  openfile['_partitions'].__getitem__(list(item))


        def __len__(self):
            with h5py.File(self.hdf5_file, 'r') as openfile:
                return  openfile['_partitions'].shape[0]

        def __str__(self):
            return "%d partitions saved on %s" %(len(self),self.hdf5_file)

    @property
    def partitions(self):
        '''Type/value of partitions is defined at time of access. If the PartitionEnsemble\
        has an associated hdf5 file (PartitionEnsemble.hdf5_file), then partitions will be \
        read and added to on the file, and not as an object in memory.'''

        if not self.hdf5_file is None:

            return PartitionEnsemble._PartitionOnFile(file=self.hdf5_file)

        else:
            return self._partitions

    @property
    def hdf5_file(self):
        '''Default location for saving/loading PartitionEnsemble if hdf5 format is used.  When this is set\
        it will automatically resave the PartitionEnsemble into the file specified.'''
        return self.hdf5_file

    @hdf5_file.setter
    def hdf5_file(self,value):
        '''Set new value for hdf5_file and automatically save to this file.'''
        self.hdf5_file=value
        self.save()


    def _check_lengths(self):
        '''
        check all state variables for equal length.  Will use length of partitions stored \
        in the hdf5 file if this is set for the PartitionEnsemble.  Otherwise just uses \
        internal lists.

        :return: boolean indicating states varaible lengths are equal

        '''
        if not self.hdf5_file is None:
            with h5py.File(self.hdf5_file) as openfile:
                if openfile['_partitions'].shape[0] == len(self.int_edges) and \
                    openfile['_partitions'].shape[0] == len(self.resolutions) and \
                    openfile['_partitions'].shape[0] == len(self.exp_edges):
                    return True
                else:
                    return False

        if self.partitions.shape[0]==len(self.int_edges) and \
                self.partitions.shape[0]==len(self.resolutions) and \
                self.partitions.shape[0]==len(self.exp_edges):
            return True
        else:
            return False


    def _combine_partitions_hdf5_files(self,otherfile):
        if self.hdf5_file is None or otherfile is None:
            raise IOError("PartitionEnsemble does not have hdf5 file currently defined")

        with h5py.File(self.hdf5_file,'a') as myfile:
            with h5py.File(otherfile,'r') as file_2_add:

                for attribute in ['_partitions', 'resolutions', 'orig_mods', "int_edges", 'exp_edges']:
                    cshape=myfile[attribute].shape
                    oshape=file_2_add[attribute].shape

                    newshape=(cshape[0]+oshape[0],cshape[1]+oshape[1])

                    myfile[attribute].resize(newshape[0],newshape[1])

                    myfile[attribute][cshape[0]:newshape[0],cshape[1]:newshape[1]]=file_2_add[attribute]

    def _append_partitions_hdf5_file(self,partitions):
        '''

        :param partitions: list of partitions (in dictionary) to add to the PartitionEnsemble.

        :type partitions: dict

        '''
        with h5py.File(self.hdf5_file,'a') as openfile:
            #Resize all of the arrays in the file
            orig_shape=openfile['_partitions'].shape
            for attribute in ['_partitions','resolutions','orig_mods',"int_edges",'exp_edges']:
                cshape=openfile[attribute].shape
                if len(cshape)==1:
                    openfile[attribute].resize( (cshape[0]+len(partitions),) )
                else:
                    openfile[attribute].resize((cshape[0] + len(partitions), cshape[1]))

            for i,part in enumerate(partitions):

                cind=orig_shape[0]+i

                #We store these on the file
                openfile['_partitions'][cind]=np.array(part['partition'])

                #We leave the new partitions only in the file.  Everything else is updated \
                # in both the PartitionEnsemble and the file

                if 'resolution' in part:
                    self.resolutions=np.append(self.resolutions,part['resolution'])
                    openfile['resolutions'][cind]=part['resolution']
                else:
                    self.resolutions=np.append(self.resolutions,None)
                    openfile['resolutions'][cind]=None


                if 'int_edges' in part:
                    self.int_edges=np.append(self.int_edges,part['int_edges'])
                    openfile['int_edges'][cind]=part['int_edges']
                else:
                    cint_edges = self.calc_internal_edges(part['partition'])
                    self.int_edges=np.append(self.int_edges,cint_edges)
                    openfile['int_edges'][cind]=cint_edges

                if 'exp_edges' in part:
                    self.exp_edges=np.append(self.exp_edges,part['exp_edges'])
                    openfile['exp_edges'][cind]=part['exp_edges']
                else:
                    cexp_edges = self.calc_expected_edges(part['partition'])
                    self.exp_edges=np.append(self.exp_edges,cexp_edges)
                    openfile['exp_edges'][cind]=cexp_edges

                if "orig_mod" in part:
                    self.orig_mods=np.append(self.orig_mods,part['orig_mod'])
                    openfile['orig_mods'][cind]=part['orig_mod']
                elif not self.resolutions[-1] is None:
                    # calculated original modularity from orig resolution
                    corigmod=self.int_edges[-1] - self.resolutions[-1] * self.exp_edges
                    self.orig_mods=np.append(self.orig_mods,corigmod)
                    openfile['orig_mods'][cind]=corigmod
                else:
                    openfile['orig_mods'][cind]=None
                    self.orig_mods=np.append(self.orig_mods,None)

            self.numparts=openfile['_partitions'].shape[0]
        assert self._check_lengths()


    def add_partitions(self,partitions,maxpt=None):
        '''
        Add additional partitions to the PartitionEnsemble object. Also adds the number of \
        communities for each.  In the case where PartitionEnsemble was openned from a file, we \
        just appended these and the other values onto each of the files.  Partitions are not kept \
        in object, however the other partitions values are.

        :param partitions: list of partitions to add to the PartitionEnsemble
        :type partitions: dict,list

        '''

        #wrap in list.
        if not hasattr(partitions,'__iter__'):
            partitions=[partitions]

        if self.hdf5_file is not None:
            # essential same as below, but everything is written to file and partitions \
            #aren't kept in object memory
            self._append_partitions_hdf5_file(partitions)



        else:
            for part in partitions:

                #This must be present
                if len(self._partitions)==0:
                    self._partitions=np.array([part['partition']])
                else:
                    self._partitions=np.append(self._partitions,[part['partition']],axis=0)

                if 'resolution' in part:
                    self.resolutions=np.append(self.resolutions,part['resolution'])
                else:
                    self.resolutions=np.append(self.resolutions,None)

                if 'int_edges' in part:
                    self.int_edges=np.append(self.int_edges,part['int_edges'])
                else:
                    cint_edges=self.calc_internal_edges(part['partition'])
                    self.int_edges=np.append(self.int_edges,cint_edges)

                if 'exp_edges' in part:
                    self.exp_edges=np.append(self.exp_edges,part['exp_edges'])
                else:
                    cexp_edges=self.calc_expected_edges(part['partition'])
                    self.exp_edges=np.append(self.exp_edges,cexp_edges)

                if "orig_mod" in part:
                    self.orig_mods=np.append(self.orig_mods,part['orig_mod'])
                elif not self.resolutions[-1] is None:
                    #calculated original modularity from orig resolution
                    self.orig_mods=np.append(self.orig_mods,self.int_edges[-1]-self.resolutions[-1]*self.exp_edges)
                else:
                    self.orig_mods=np.append(self.orig_mods,None)



                self.numcoms=np.append(self.numcoms,get_number_of_communities(part['partition'],
                                                              min_com_size=self.min_com_size))

                assert self._check_lengths()
                self.numparts=len(self.partitions)
            #update the pruned set
        self.apply_CHAMP(maxpt=self.maxpt)



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

    def merge_ensemble(self,otherEnsemble,new=True):
        '''
        Combine to PartitionEnsembles.  Checks for concordance in the number of vertices. \
        Assumes that internal ordering on the graph nodes for each is the same.

        :param otherEnsemble: otherEnsemble to merge
        :param new: create a new PartitionEnsemble object? Otherwise partitions will be loaded into \
        the one of the original partition ensemble objects (the one with more partitions in the first place).
        :type new: bool
        :return:  PartitionEnsemble reference with merged set of partitions

        '''

        if not self.graph.vcount()==otherEnsemble.graph.vcount():
            raise ValueError("PartitionEnsemble graph vertex counts do not match")

        if new:
            bothpartitions=self.get_partition_dictionary()+otherEnsemble.get_partition_dictionary()
            return PartitionEnsemble(self.graph,listofparts=bothpartitions)

        else:
            if self.numparts<otherEnsemble.numparts:
                #reverse order of merging
                return otherEnsemble.merge_ensemble(self,new=False)
            else:
                if not self.hdf5_file is None and not otherEnsemble.hdf5_file is None:
                    #merge the second hdf5_file onto the other and then reopen it to
                    #reload everything.
                    self._combine_partitions_hdf5_files(otherEnsemble.hdf5_file)
                    self.open(self.hdf5_file)
                    return self
                else:
                    self.add_partitions(otherEnsemble.get_partition_dictionary())
                    return self

    def get_coefficient_array(self):
        '''
        Create array of coefficents for each partition.

        :return: np.array with coefficents for each of the partions

        '''

        outlist=np.array([[self.int_edges[0],
                self.exp_edges[0]]])

        for i in range(1,self.numparts):
            outlist=np.append(
                outlist,[[
                self.int_edges[i],
                self.exp_edges[i]
            ]],axis=0)

        return outlist

    @property
    def unique_coeff_indices(self):
        if self._uniq_coeff_indices is None:
            self._uniq_coeff_indices=self.get_unique_coeff_indices()
            return self._uniq_coeff_indices

    @property
    def unique_partition_indices(self):
        if self._uniq_partition_indices is None:
            self._twin_partitions,self._uniq_partition_indices=self._get_unique_twins_and_partition_indices()
        return self._uniq_partition_indices

    @property
    def twin_partitions(self):
        '''
        We define twin partitions as those that have the same coefficients but are different partitions.\
        To find these we look for the diffence in the partitions with the same coefficients.

        :return: List of groups of the indices of partitions that have the same coefficient but \
        are non-identical.
        :rtype: list of list (possibly empty if no twins)
        '''

        if self._twin_partitions is None:
            self._twin_partitions,self._uniq_partition_indices=self._get_unique_twins_and_partition_indices()
        return self._twin_partitions





    def get_unique_coeff_indices(self):
        ''' Get the indices for the partitions with unique coefficient \
         :math:`\\hat{A}=\\sum_{ij}A_{ij}` \
         :math:`\\hat{P}=\\sum_{ij}P_{ij}`

         Note that for each replicated partition we return the index of one (the earliest in the list) \
         the replicated

        :return: the indices of unique coeficients
        :rtype: np.array
        '''
        _,indices=np.unique(self.get_coefficient_array(),return_index=True,axis=0)
        return indices

    def _reindex_part_array(self,part_array):
        '''we renumber partitions labels to ensure that each goes from number 0,1,2,...
        in order to compare.

        :param part_array:
        :type part_array:
        :return: relabeled array
        :rtype:
        '''
        out_array=np.zeros(part_array.shape)
        for i in range(part_array.shape[0]):
            clabdict={}
            for j in range(part_array.shape[1]):
                #Use len of cdict as value
                clabdict[part_array[i][j]]=clabdict.get(part_array[i][j],len(clabdict))

            for j in range(part_array.shape[1]):
                out_array[i][j]=clabdict[part_array[i][j]]

        return out_array


    def _get_unique_twins_and_partition_indices(self, reindex=True):
        '''
        Returns the (possibly empty) list of twin partitions and the list of unique partitions.

        :param reindex: if True, will reindex partitions that it is comparing to ensure they are unique under \
        permutation.
        :return: list of twin partition (can be empty), list of indicies of unique partitions.
        :rtype: list,np.array
        '''
        uniq,index,reverse,counts=np.unique(self.get_coefficient_array(),
                                            return_index=True,return_counts=True,
                                            return_inverse=True,axis=0)

        ind2keep=index[np.where(counts==1)[0]]
        twin_inds=[]

        for ind in np.where(counts>1)[0]:
            #we have to load the partitions and compare them to each other
            revinds=np.where(reverse==ind)[0]
            parts2comp=self.partitions[np.where(reverse==ind)[0]]
            if reindex:
                reindexed_parts2comp=self._reindex_part_array(parts2comp)
            else:
                reindexed_parts2comp=parts2comp

            #here curpart inds is which of of this current group of partitions are unique
            _,curpart_inds=np.unique(reindexed_parts2comp,axis=0,return_index=True)
            #len of curpart_inds determines how many of the current ind group get added to
            #the ind2keep.  should always be at least one.
            if len(curpart_inds)>1: #matching partitions with different coeffs
                twin_inds.append(revinds[curpart_inds])
            ind2keep=np.append(ind2keep,revinds[curpart_inds])

        np.sort(ind2keep)
        return  twin_inds,ind2keep

    def get_unique_partition_indices(self,reindex=True):
        '''
       This returns the indices for the partitions who are unique.  This could be larger than the
       indices for the unique coeficient since multiple partitions can give rise to the same coefficient. \
       In practice this has been very rare.  This function can take sometime for larger network with many \
       partitions since it reindex the partitions labels to ensure they aren't permutations of each other.

       :param reindex: if True, will reindex partitions that it is comparing to ensure they are unique under \
       permutation.
       :return: list of twin partition (can be empty), list of indicies of unique partitions.
       :rtype: list,np.array
       '''
        _,uniq_inds=self._get_unique_twins_and_partition_indices(reindex=reindex)
        return uniq_inds



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

    def _write_graph_to_hd5f_file(self,file,compress=4):
        '''
        Write the internal graph to hd5f file saving the edge lists, the edge properties, and the \
        vertex properties all as subgroups.  We only save the edges, and the vertex and node attributes

        :param file: openned h5py.File
        :type file: h5py.File
        :return: reference to the File
        '''

        # write over previous if exists.
        if 'graph' in file.keys():
            del file['graph']

        grph=file.create_group("graph")

        grph.create_dataset('directed',data=int(self.graph.is_directed()))
        #save edge list as graph.ecount x 2 numpy array
        grph.create_dataset("edge_list",
                            data=np.array([e.tuple for e in self.graph.es]),compression="gzip",compression_opts=compress)

        edge_atts=grph.create_group('edge_attributes')
        for attrib in self.graph.edge_attributes():

            edge_atts.create_dataset(attrib,
                                     data=np.array(self.graph.es[attrib]),compression="gzip",compression_opts=compress)

        node_atts=grph.create_group("node_attributes")
        for attrib in self.graph.vertex_attributes():
            node_atts.create_dataset(attrib,
                                     data=np.array(self.graph.vs[attrib]),compression="gzip",compression_opts=compress)
        return file

    def _read_graph_from_hd5f_file(self,file):
        '''
        Load self.graph from hd5f file.  Sets self.graph as new igraph created from edge list \
        and attributes stored in the file.

        :param file: Opened hd5f file that contains the edge list, edge attributes, and \
        node attributes stored in the hierarchy as PartitionEnsemble._write_graph_to_hd5f_file.

        :type file: h5py.File

        '''



        grph=file['graph']
        directed=bool(grph['directed'].value)
        self.graph=ig.Graph().TupleList(grph['edge_list'],directed=directed)
        for attrib in grph['edge_attributes'].keys():
            self.graph.es[attrib]=grph['edge_attributes'][attrib][:]
        for attrib in grph['node_attributes'].keys():
            self.graph.vs[attrib] = grph['node_attributes'][attrib][:]


    def save(self,filename=None,dir=".",hdf5=None,compress=9):
        '''
        Use pickle or h5py to store representation of PartitionEnsemble in compressed file.  When called \
        if object has an assocated hdf5_file, this is the default file written to.  Otherwise objected \
        is stored using pickle.

        :param filename: name of file to write to.  Default is created from name of ParititonEnsemble\: \
            "%s_PartEnsemble_%d" %(self.name,self.numparts)
        :param hdf5: save the PartitionEnsemble object as a hdf5 file.  This is \
        very useful for larger partition sets, especially when you only need to work \
        with the optimal subset.  If object has hdf5_file attribute saved \
        this becomes the default
        :type hdf5: bool
        :param compress: Level of compression for partitions in hdf5 file.  With less compression, files take \
        longer to write but take up more space.  9 is default.
        :type compress: int [0,9]
        :param dir: directory to save graph in.  relative or absolute path.  default is working dir.
        :type dir: str
        '''

        if hdf5 is None:
            if self.hdf5_file is None:
                hdf5 is False
            else:
                hdf5 is True



        if filename is None:
            if hdf5:
                if self.hdf5_file is None:
                    filename="%s_PartEnsemble_%d.hdf5" %(self.name,self.numparts)
                else:
                    filename=self.hdf5_file
            else:
                filename="%s_PartEnsemble_%d.gz" %(self.name,self.numparts)

        if hdf5:
            with h5py.File(filename,'w') as outfile:

                for k,val in iteritems(self.__dict__):
                    #store dictionary type object as its own group
                    if k=='graph':
                        self._write_graph_to_hd5f_file(outfile,compress=compress)

                    elif isinstance(val,dict):
                        indgrp=outfile.create_group(k)
                        for ind,dom in iteritems(val):
                            indgrp.create_dataset(str(ind),data=dom,compression="gzip",compression_opts=compress)

                    elif isinstance(val,str):
                        outfile.create_dataset(k,data=val)

                    elif hasattr(val,"__len__"):
                        data=np.array(val)

                        #1D array don't have a second shape index (ie np.array.shape[1] can throw \
                        #IndexError
                        cshape=list(data.shape)
                        cshape[0]=None
                        cshape=tuple(cshape)

                        cdset = outfile.create_dataset(k, data=data, maxshape=cshape,
                                                       compression="gzip",compression_opts=compress)

                    elif not val is None:
                            #Single value attributes
                            cdset = outfile.create_dataset(k,data=val)


            self.hdf5_file=filename

        else:
            with gzip.open(os.path.join(dir,filename),'wb') as fh:
                pickle.dump(self,fh)

    def save_graph(self,filename=None,dir="."):
        '''
        Save a copy of the graph with each of the optimal partitions stored as vertex attributes \
        in graphml compressed format.  Each partition is attribute names part_gamma where gamma is \
        the beginning of the partitions domain of dominance

        :param filename: name of file to write out to.  Default is self.name.graphml.gz or \
        :type filename: str
        :param dir: directory to save graph in.  relative or absolute path.  default is working dir.
        :type dir: str
        '''

        #TODO add other graph formats for saving.
        if filename is None:
            filename=self.name+".graphml.gz"
        outgraph=self.graph.copy()
        #Add the CHAMP partitions to the outgraph
        for ind in self.get_CHAMP_indices():
            part_name="part_%.3f" %(self.ind2doms[ind][0][0])
            outgraph.vs[part_name]=self.partitions[ind]

        outgraph.write_graphmlz(os.path.join(dir,filename))



    def open(self,filename):
        '''
        Loads pickled PartitionEnsemble from file.

        :param file:  filename of pickled PartitionEnsemble Object

        :return: writes over current instance and returns the reference

        '''

        #try openning it as an hd5file
        try:
            with h5py.File(filename,'r') as infile:
                self._read_graph_from_hd5f_file(infile)
                for key in infile.keys():
                    if key!='graph' and key!='_partitions':
                        #get domain indices recreate ind2dom dict
                        if key=='ind2doms':
                            self.ind2doms={}
                            for ind in infile[key]:
                                self.ind2doms[int(ind)]=infile[key][ind][:]
                        else:
                            try:
                                self.__dict__[key]=infile[key][:]
                            except ValueError:
                                self.__dict__[key]=infile[key].value

            #store this for accessing partitions

            self.hdf5_file=filename
            return self

        except IOError:

            with gzip.open(filename,'rb') as fh:
                opened=pickle.load(fh)

            openedparts=opened.get_partition_dictionary()

            #construct and return
            self.__init__(opened.graph,listofparts=openedparts)
            return self


    def _sub_tex(self,str):
        new_str = re.sub("\$", "", str)
        new_str = re.sub("\\\\ge", ">=", new_str)
        new_str = re.sub("\\\\", "", new_str)
        return new_str

    def _remove_tex_legend(self,legend):
        for text in legend.get_texts():
            text.set_text(self._sub_tex(text.get_text()))
        return legend

    def _remove_tex_axes(self, axes):
        axes.set_title(self._sub_tex(axes.get_title()))
        axes.set_xlabel(self._sub_tex(axes.get_xlabel()))
        axes.set_ylabel(self._sub_tex(axes.get_ylabel()))
        return axes

    def plot_modularity_mapping(self,ax=None,downsample=2000,champ_only=False,legend=True,
                                no_tex=False):
        '''

        Plot a scatter of the original modularity vs gamma with the modularity envelope super imposed. \
        Along with communities vs :math:`\\gamma` on a twin axis.  If no orig_mod values are stored in the \
        ensemble, just the modularity envelope is plotted.  Depending on the backend used to render plot \
        the latex in the labels can cause error.  If you are getting RunTime errors when showing or saving \
        the plot, try setting no_tex=True

        :param ax: axes to draw the figure on.
        :type ax: matplotlib.Axes
        :param champ_only: Only plot the modularity envelop represented by the CHAMP identified subset.
        :type champ_only: bool
        :param downsample: for large number of runs, we down sample the scatter for the number of communities \
        and the original partition set.  Default is 2000 randomly selected partitions.
        :type downsample: int
        :param legend: Add legend to the figure.  Default is true
        :type legend: bool
        :param no_tex: Use latex in the legends.  Default is true.  If error is thrown on plotting try setting \
        this to false.
        :type no_tex: bool
        :return: axes drawn upon
        :rtype: matplotlib.Axes


        '''

        if ax == None:
            f = plt.figure()
            ax = f.add_subplot(111)

        if not no_tex:
            rc('text',usetex=True)
        else:
            rc('text',usetex=False)



        # check for downsampling and subset indices
        if downsample and downsample<=len(self.partitions):
            rand_ind=np.random.choice(range(len(self.partitions)),size=downsample)
        else:
            rand_ind=range(len(self.partitions))

        allgams = [self.resolutions[ind] for ind in rand_ind]
        allcoms = [self.numcoms[ind] for ind in rand_ind]

        if not champ_only and not self.orig_mods[0] is None :


            allmods=[self.orig_mods[ind] for ind in rand_ind]

            ax.set_ylim([np.min(allmods) - 100, np.max(allmods) + 100])
            mk1 = ax.scatter(allgams, allmods,
                             color='red', marker='.', alpha=.6, s=10,
                             label="modularity", zorder=2)

        #take the x-coord of first point in each domain

        #Get lists for the champ subset
        champ_inds=self.get_CHAMP_indices()

        # take the x-coord of first point in each domain
        gammas=[ self.ind2doms[ind][0][0] for ind in champ_inds  ]
        # take the y-coord of first point in each domain
        mods = [self.ind2doms[ind][0][1] for ind in champ_inds]

        champ_coms = [self.numcoms[ind] for ind in champ_inds]



        mk5 = ax.plot(gammas, mods, ls='--', color='green', lw=3, zorder=3)
        mk5 = mlines.Line2D([], [], color='green', ls='--', lw=3)

        mk2 = ax.scatter(gammas, mods, marker="v", color='blue', s=60, zorder=4)
        #     ax.scatter(gamma_ins,orig_mods,marker='x',color='red')
        ax.set_ylabel("modularity")



        a2 = ax.twinx()
        a2.grid('off')
        #     a2.scatter(allgammas,allcoms,marker="^",color="#fe9600",alpha=1,label=r'\# communities ($\ge 5$ nodes)',zorder=1)

        sct2 = a2.scatter(allgams, allcoms, marker="^", color="#91AEC1",
                          alpha=1, label=r'\# communities ($\ge %d$ nodes)'%(self.min_com_size),
                          zorder=1)
        #     sct2.set_path_effects([path_effects.SimplePatchShadow(alpha=.5),path_effects.Normal()])

        # fake for legend with larger marker size
        mk3 = a2.scatter([], [], marker="^", color="#91AEC1", alpha=1,
                         label=r'\# communities ($\ge %d$)'%(self.min_com_size),
                         zorder=1,
                         s=20)



        stp = a2.step(gammas, champ_coms, color="#004F2D", where='post',
                      path_effects=[path_effects.SimpleLineShadow(alpha=.5), path_effects.Normal()])
        #     stp.set_path_effects([patheffects.Stroke(linewidth=1, foreground='black'),
        #                     patheffects.Normal()])

        # for legend
        mk4 = mlines.Line2D([], [], color='#004F2D', lw=2,
                            path_effects=[path_effects.SimpleLineShadow(alpha=.5), path_effects.Normal()])


        a2.set_ylabel(r"\# communities ($\ge 5$ nodes)")

        ax.set_zorder(a2.get_zorder() + 1)  # put ax in front of ax2
        ax.patch.set_visible(False)  # hide the 'canvas'

        ax.set_xlim(xmin=0, xmax=max(allgams))
        if legend:
            l = ax.legend([mk1, mk3, mk2, mk4, mk5],
                          ['modularity', r'\# communities ($\ge %d $ nodes)' %(self.min_com_size), "transitions,$\gamma$",
                           r"\# communities ($\ge %d$ nodes) optimal" %(self.min_com_size), "convex hull of $Q(\gamma)$"],
                          bbox_to_anchor=[0.5, .87], loc='center',
                          frameon=True, fontsize=14)
            l.get_frame().set_fill(False)
            l.get_frame().set_ec("k")
            l.get_frame().set_linewidth(1)
            if no_tex:
                l=self._remove_tex_legend(l)

        if no_tex: #clean up the tex the axes
            a2=self._remove_tex_axes(a2)
            ax=self._remove_tex_axes(ax)

        return ax

            


class MultiLayerPartitionEnsemble():

    '''
    MultilayerPartitionEnsemble.  Similar to louvain_ext.PartitionEnsemble except that \
    multigraphs are stored as supradjacency format (igraph does't have a good multigraph \
    representation).
    '''

    def __init__(self,adj_A=None,adj_P=None,adj_C=None,partitions=None,coef_array=None,maxpt=None,name="unnamed_multilayer_ensemble"):
        self.adj_A=adj_A
        self.adj_C=adj_C
        self.adj_P=adj_P
        self.partitions=partitions
        self.maxpt=maxpt
        self.name=name
        self.numparts=0 if self.partitions is None else len((self.partitions))

        if coef_array is None and not self.partitions is None:
            self.coef_array=create_coefarray_from_partitions(self.adj_A,
                                                             self.adj_P,self.adj_C,nprocesses=cpu_count())
        self.ind2doms=self.apply_CHAMP(maxpt=self.maxpt)

    def apply_CHAMP(self,maxpt=None):
        '''
        Apply champ internally to the coefficent array.

        :param maxpt: Cut off value for inclusion of domains in both the :math:`\\gamma` and :math:`\\omega` axes.
        :type maxpt: (float,float)
        '''

        if maxpt is None:
            maxpt=self.maxpt
        if self.coef_array is None:
            create_coefarray_from_partitions(self.partitions,
                                             self.adj_A,self.adj_P,self.adj_C,nprocesses=cpu_count())
        self.ind2doms=get_intersection(self.coef_array,maxpt)

    def reset_min_com_size(self,new_min_size):
        '''
        Result the minimum community size used to define the total number of communities for each partition \
        Goes back through and recalculates community sizes for each partition.

        :param new_min_size: new minimum size of communities
        :type new_min_size: int

        '''
        for i,part in enumerate(self.partitions):
            self.numcoms[i]=get_number_of_communities(part,new_min_size)

    def save(self,filename=None):
        '''
        Use pickle to dump representation to compressed file

        :param filename:

        '''
        if filename is None:
            filename="%s_MultiPartEnsemble_%d.gz" %(self.name,self.numparts)

        with gzip.open(filename,'wb') as fh:
            pickle.dump(self,fh)


    def open(filename):
        '''
        Loads pickled PartitionEnsemble from file.

        :param file:  filename of pickled PartitionEnsemble Object
        :return: writes over current instance and returns the reference

        '''
        with gzip.open(filename,'rb') as fh:
            opened=pickle.load(fh)

        #construct and return
        return MultiLayerPartitionEnsemble(opened.adj_A,opened.adj_P,opened.adj_C,
                                           partitions=opened.partitions,coef_array=opened.coef_array,
                                           maxpt=opened.maxpt,name=opened.name)

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

def get_number_of_communities(partition,min_com_size=0):
    '''

    :param partition: list with community assignments (labels must be of hashable type \
    e.g. int,string, etc...).
    :type partition: list
    :param min_com_size: Minimum size to include community in the total number of communities (default is 0)
    :type min_com_size: int
    :return: number of communities
    :rtype: int
    '''
    count_dict={}
    for label in partition:
        count_dict[label]=count_dict.get(label,0)+1

    tot_coms=0
    for k,val in iteritems(count_dict):
        if val>=min_com_size:
            tot_coms+=1
    return tot_coms

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

        svec=np.array(lmap(lambda x :strengths[x],subg.vs['id']))
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
    new_member=[-1 for i in range(len(rev_order))]

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
    for i in range(nruns):
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
            print("Run %d at gamma = %.3f.  Return time: %.4f" %(progress,gamma,time()-t))

    return outparts
def parallel_louvain(graph,start=0,fin=1,numruns=200,maxpt=None,
                     numprocesses=None, attribute=None,weight=None,node_subset=None,progress=False):
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
    for i in range(numruns):
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