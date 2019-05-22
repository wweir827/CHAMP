#Py 2/3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division # use // to specify int div.


from .champ_functions import get_expected_edges
from .champ_functions import get_expected_edges_ml
from .champ_functions import get_sum_internal_edges
from .PartitionEnsemble import PartitionEnsemble

import sys, os
import tempfile
from contextlib import contextmanager
from multiprocessing import Pool,cpu_count
import itertools
import igraph as ig
import louvain
import numpy as np
import tqdm
from time import time
import warnings
import logging
#logging.basicConfig(format=':%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)
logging.basicConfig(format=':%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)


iswin = os.name == 'nt'
is_py3 = sys.version_info >= (3, 0)


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


##### STATIC METHODS FOR LOUVAIN EXT######

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

def permute_vector(rev_order, membership):
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

def permute_memvec(permutation,membership):
	outvec=np.array([-1 for _ in range(len(membership))])
	for i,val in enumerate(permutation):
		outvec[val]=membership[i]

	return outvec

def run_louvain_windows(graph,gamma,nruns,weight=None,node_subset=None,attribute=None,output_dictionary=False):
	'''
	Call the louvain method with igraph as input directly.  This is needed for windows system\
	because tmp files cannot be closed and reopened

	This takes as input a graph file (instead of the graph object) to avoid duplicating
	references in the context of parallelization.  To allow for flexibility, it allows for
	subsetting of the nodes each time.

	:param graph: igraph
	:param node_subset: Subeset of nodes to keep (either the indices or list of attributes)
	:param gamma: resolution parameter to run louvain
	:param nruns: number of runs to conduct
	:param weight: optional name of weight attribute for the edges if network is weighted.
	:param output_dictionary: Boolean - output a dictionary representation without attached graph.
	:return: list of partition objects

	'''

	np.random.seed() #reset seed for each process

	#Load the graph from the file
	g = graph
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

	if weight is True:
		weight='weight'


	outparts=[]
	for i in range(nruns):
		rand_perm = list(np.random.permutation(g.vcount()))
		rperm = rev_perm(rand_perm)
		gr=g.permute_vertices(rand_perm) #This is just a labelling switch.  internal properties maintined.

		#In louvain > 0.6, change in the way the different methods are called.
		#modpart=louvain.RBConfigurationVertexPartition(gr,resolution_parameter=gamma)
		rp = louvain.find_partition(gr,louvain.RBConfigurationVertexPartition,weights=weight,
									resolution_parameter=gamma)

		#store the coefficients in return object.
		A=get_sum_internal_edges(rp,weight)
		P=get_expected_edges(rp,weight,directed=g.is_directed())


		outparts.append({'partition': permute_vector(rperm, rp.membership),
						 'resolution':gamma,
						 'orig_mod': rp.quality(),
						 'int_edges':A,
						 'exp_edges':P})

	if not output_dictionary:
		return PartitionEnsemble(graph=g,listofparts=outparts)
	else:
		return outparts
	return part_ensemble


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

	if weight is True:
		weight='weight'


	outparts=[]
	for i in range(nruns):
		rand_perm = list(np.random.permutation(g.vcount()))
		rperm = rev_perm(rand_perm)
		gr=g.permute_vertices(rand_perm) #This is just a labelling switch.  internal properties maintined.

		#In louvain > 0.6, change in the way the different methods are called.
		#modpart=louvain.RBConfigurationVertexPartition(gr,resolution_parameter=gamma)
		rp = louvain.find_partition(gr,louvain.RBConfigurationVertexPartition,weights=weight,
									resolution_parameter=gamma)

		#old way of calling
		# rp = louvain.find_partition(gr, method='RBConfiguration',weight=weight,  resolution_parameter=gamma)

		#store the coefficients in return object.
		A=get_sum_internal_edges(rp,weight)
		P=get_expected_edges(rp,weight,directed=g.is_directed())


		outparts.append({'partition': permute_vector(rperm, rp.membership),
						 'resolution':gamma,
						 'orig_mod': rp.quality(),
						 'int_edges':A,
						 'exp_edges':P})

	if not output_dictionary:
		return PartitionEnsemble(graph=g,listofparts=outparts)
	else:
		return outparts
	return part_ensemble





def _run_louvain_parallel(gfile_gamma_nruns_weight_subset_attribute):
	'''
	Parallel wrapper with single argument input for calling :meth:`louvain_ext.run_louvain`

	:param gfile_att_2_id_dict_shared_gamma_runs_weight: tuple or list of arguments to supply
	:returns: PartitionEnsemble of graph stored in gfile
	'''
	#unpack
	gfile,gamma,nruns,weight,node_subset,attribute=gfile_gamma_nruns_weight_subset_attribute
	t=time()
	outparts=run_louvain(gfile,gamma,nruns=nruns,weight=weight,node_subset=node_subset,attribute=attribute,output_dictionary=True)

	# if progress is not None:
	# 	if progress%update==0:
	# 		print("Run %d at gamma = %.3f.  Return time: %.4f" %(progress,gamma,time()-t))

	return outparts

def parallel_louvain(graph,start=0,fin=1,numruns=200,maxpt=None,
					 numprocesses=None, attribute=None,weight=None,node_subset=None,progress=None):
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
	:param progress:  Print progress in parallel execution every `n` iterations.
	:return: PartitionEnsemble of all partitions identified.

	'''

	if iswin: #on a windows system
		warnings.warn("Parallel Louvain function is not available of windows system.  Running in serial",
					  UserWarning)
		for i,gam in enumerate(np.linspace(start,fin,numruns)):
			cpart_ens=run_louvain_windows(graph=graph,nruns=1,gamma=gam,node_subset=node_subset,
										attribute=attribute,weight=weight)
			if i==0:
				outpart_ens=cpart_ens
			else:
				outpart_ens=outpart_ens.merge_ensemble(cpart_ens,new=False) #merge current run with new
		return outpart_ens

	parallel_args=[]
	if numprocesses is None:
		numprocesses=cpu_count()

	if weight is True:
		weight='weight'

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
		curg = start + ((fin - start) / (1.0 * numruns)) * i
		parallel_args.append((graphfile, curg, 1, weight, None, None))

	parts_list_of_list=[]
	# use a context manager so pools properly shut down
	with terminating(Pool(processes=numprocesses)) as pool:

		if progress:
			tot = len(parallel_args)
			with tqdm.tqdm(total=tot) as pbar:
				# parts_list_of_list=pool.imap(_parallel_run_louvain_multimodularity,args)
				for i, res in tqdm.tqdm(enumerate(pool.imap(_run_louvain_parallel, parallel_args)), miniters=tot):
					# if i % 100==0:
					pbar.update()
					parts_list_of_list.append(res)
		else:
			parts_list_of_list=pool.map(_run_louvain_parallel, parallel_args )

	#for debugging
	# parts_list_of_list=list(map(_run_louvain_parallel,parallel_args))

	all_part_dicts=[pt for partrun in parts_list_of_list for pt in partrun]
	tempf.close()
	outensemble=PartitionEnsemble(graph,listofparts=all_part_dicts,maxpt=maxpt)
	return outensemble



#### MULTI-LAYER Louvain static methods

#MUTLILAYER GRAPH CREATION

def _create_interslice(interlayer_edges, layer_vec, directed=False):
	"""


	"""
	weights=[]
	layers = np.unique(layer_vec)
	layer_edges = set()
	for e in interlayer_edges:
		ei,ej=e[0],e[1]
		lay_i = layer_vec[ei]
		lay_j = layer_vec[ej]
		if len(e)>2:
			weights.append(e[2])
		assert lay_i != lay_j #these shoudl be interlayer edges
		if lay_i < lay_j:
			layer_edges.add((lay_i, lay_j))
		else:
			layer_edges.add((lay_j, lay_i))


	slice_couplings = ig.Graph(n=len(layers), edges=list(layer_edges), directed=directed)
	if len(weights) == 0:
		weights=1
	slice_couplings.es['weight']=weights
	return slice_couplings


def _create_multilayer_igraphs_from_super_adj_igraph(intralayer_igraph,layer_vec):
	"""
	For falling back on the normal louvain method.  We use the single layer intralayer_igraph to\
	 create igraph representations for each of the layers.
	:param intralayer_igraph: igraph.Graph super_adjacency representation
	:param layer_vec: indicator of which layer each node is in.
	:return:
	"""

	adj=np.array(intralayer_igraph.get_adjacency().data)
	layer_vals = np.unique(layer_vec)

	layers=[]
	# We calculate the induced subgraph for each layer by identifying all of the edges
	# in that layer and creating a new igraph object for each (without deleting vertices)
	for layer in layer_vals:
		#
		cinds=np.where(layer_vec==layer)[0]
		cedges=set()
		for v in cinds:
			cedges.update(intralayer_igraph.incident(v))

		cedges=list(cedges)
		cgraph=intralayer_igraph.subgraph_edges(edges=cedges,delete_vertices=False)
		layers.append(cgraph)
	return layers


def _create_all_layers_single_igraph(intralayer_edges, layer_vec, directed=False):
	"""
	"""
	#create a single igraph
	layers, cnts = np.unique(layer_vec, return_counts=True)
	layer_elists = []
	layer_weights=[ ]
	# we divide up the edges by layer
	for e in intralayer_edges:
		ei,ej=e[0],e[1]
		if not directed: #switch order to preserve uniqness
			if ei>ej:
				ei,ej=e[1],e[0]

		layer_elists.append((ei, ej))
		if len(e)>2:
			layer_weights.append(e[2])

	layer_graphs = []
	cgraph = ig.Graph(n=len(layer_vec), edges=layer_elists, directed=directed)
	if len(layer_weights) > 0:  # attempt to set the intralayer weights
		cgraph.es['weight'] = layer_weights
	else:
		cgraph.es['weight']=1
	cgraph.vs['nid']=range(cgraph.vcount())
	cgraph.vs['layer_vec']=layer_vec
	return cgraph
	# layer_graphs.append(cgraph)
	# return layer_graphs

def _create_all_layer_igraphs_multi(intralayer_edges, layer_vec, directed=False):
	"""
	"""

	layers, cnts = np.unique(layer_vec, return_counts=True)
	layer_elists = [[] for _ in range(len(layers))]
	layer_weights=[[] for _ in range(len(layers))]
	# we divide up the edges by layer
	for e in intralayer_edges:
		ei,ej=e[0],e[1]
		if not directed: #switch order to preserve uniqness
			if ei>ej:
				ei,ej=e[1],e[0]

		# these should all be intralayer edges
		lay_i, lay_j = layer_vec[ei], layer_vec[ej]
		assert lay_i == lay_j

		coffset=np.sum(cnts[:lay_i])#indexing for edges must start with 0 for igraph

		layer_elists[lay_i].append((ei-coffset, ej-coffset))
		if len(e)>2:
			layer_weights[lay_i].append(e[2])

	layer_graphs = []
	tot = 0
	for i, layer_elist in enumerate(layer_elists):
		if not directed:
			layer_elist=list(set(layer_elist)) #prune out non-unique
		#you have adjust the elist to start with 0 for first node
		cnts[i]
		cgraph = ig.Graph(n=cnts[i], edges=layer_elist, directed=directed)
		assert cgraph.vcount()==cnts[i],'edges indicated more nodes within graph than the layer_vec'
		cgraph.vs['nid'] = range(tot , tot +cnts[i])  # each node in each layer gets a unique id
		if len(layer_weights[i])>0: #attempt to set the intralayer weights
			cgraph.es['weight']=layer_weights[i]
		tot += cnts[i]
		layer_graphs.append(cgraph)

	return layer_graphs


def _label_nodes_by_identity(intralayer_graphs, interlayer_edges, layer_vec):
	"""Go through each of the nodes and determine which ones are shared across multiple slices.\
	We create an attribute on each of the graphs to indicate the shared identity \
	of that node.  This is done through tracking the predecessors of the node vi the interlayer\
	connections

	"""

	namedict = {}
	backedges = {}

	# For each node we hash if it has any neighbors in the layers behind it.

	for e in interlayer_edges:
		ei,ej=e[0],e[1]
		if ei < ej:
			backedges[ej] = backedges.get(ej, []) + [ei]
		else:
			backedges[ei] = backedges.get(ei, []) + [ej]

	offset = 0  # duplicate names used
	for i, lay in enumerate(layer_vec):

		if i not in backedges:  # node doesn't have a predecessor
			namedict[i] = i - offset
		else:
			pred = backedges[i][0] #get one of the predecessors
			namedict[i] = namedict[pred]  # get the id of the predecessor
			offset += 1

	for graph in intralayer_graphs:
		graph.vs['shared_id'] = list(map(lambda x: namedict[x], graph.vs['nid']))
		assert len(set(graph.vs['shared_id']))==len(graph.vs['shared_id']), "IDs within a slice must all be unique"


def create_multilayer_igraph_from_edgelist(intralayer_edges, interlayer_edges, layer_vec, inter_directed=False,
										   intra_directed=False):
	"""
	   We create an igraph representation used by the louvain package to represents multi-slice graphs.  \
	   For this method only two graphs are created :
	   intralayer_graph : all edges withis this graph are treated equally though the null model is adjusted \
	   based on each slice's degree distribution
	   interlayer_graph:  single graph that contains only interlayer connections between all nodes

	:param intralayer_edges: edges representing intralayer connections.  Note each node should be assigned a unique\
	index.
	:param interlayer_edges: connection across layers.
	:param layer_vec: indication of which layer each node is in.  This important in computing the modulary modularity\
	null model.
	:param directed: If the network is directed or not
	:return: intralayer_graph,interlayer_graph
	"""
	t=time()
	interlayer_graph = _create_all_layers_single_igraph(interlayer_edges, layer_vec=layer_vec, directed=inter_directed)
	# interlayer_graph=interlayer_graph[0]
	logging.debug("create interlayer : {:.4f}".format(time()-t))
	t=time()
	intralayer_graph = _create_all_layers_single_igraph(intralayer_edges, layer_vec, directed=intra_directed)
	logging.debug("create intrallayer : {:.4f}".format(time()-t))
	t=time()
	return intralayer_graph,interlayer_graph


def call_slices_to_layers_from_edge_list(intralayer_edges, interlayer_edges, layer_vec, directed=False):
	"""
	   We create an igraph representation used by the louvain package to represents multi-slice graphs.  This returns \
	   three values:
		layers : list of igraphs each one representing a single slice in the network (all nodes across all layers \
		are present but only the edges in that slice)
		interslice_layer: igraph representing interlayer connectiosn
		G_full : igraph with connections for both inter and intra slice connections across all nodes ( differentiated) \
		by igraph.es attribute.

	:param intralayer_edges:
	:param interlayer_edges:
	:param layer_vec:
	:param directed:
	:return: layers
	"""
	t=time()
	interlayer_graph = _create_interslice(interlayer_edges,layer_vec=layer_vec, directed=directed)
	# interlayer_graph=interlayer_graph[0]
	logging.debug("create interlayer : {:.4f}".format(time()-t))
	t=time()
	intralayer_graphs = _create_all_layer_igraphs_multi(intralayer_edges, layer_vec, directed=directed)
	logging.debug("create intrallayer : {:.4f}".format(time()-t))
	t=time()

	_label_nodes_by_identity(intralayer_graphs, interlayer_edges, layer_vec)
	logging.debug("label nodes : {:.4f}".format(time()-t))
	t=time()
	interlayer_graph.vs['slice'] = intralayer_graphs
	layers, interslice_layer, G_full = louvain.slices_to_layers(interlayer_graph, vertex_id_attr='shared_id')
	logging.debug("louvain call : {:.4f}".format(time()-t))
	t=time()
	return layers, interslice_layer, G_full

def adjacency_to_edges(A):
	nnz_inds = np.nonzero(A)
	nnzvals = np.array(A[nnz_inds])
	if len(nnzvals.shape)>1:
		nnzvals=nnzvals[0] #handle scipy sparse types
	return list(zip(nnz_inds[0], nnz_inds[1], nnzvals))


def create_multilayer_igraph_from_adjacency(A,C,layer_vec,inter_directed=False,intra_directed=False):
	"""
	Create the multilayer igraph representation necessary to call igraph-louvain \
	in the multilayer context.  Edge list are formed and champ_fucntions.create_multilayer_igraph_from_edgelist \
	is called.  Each edge list includes the weight of the edge \
	as indicated in the appropriate adjacency matrix.

	:param A:
	:param C:
	:param layer_vec:
	:return:
	"""

	nnz_inds = np.nonzero(A)
	nnzvals = np.array(A[nnz_inds])
	if len(nnzvals.shape)>1:
		nnzvals=nnzvals[0] #handle scipy sparse types

	intra_edgelist = adjacency_to_edges(A)
	inter_edgelist = adjacency_to_edges(C)


	return create_multilayer_igraph_from_edgelist(intralayer_edges=intra_edgelist,
												  interlayer_edges=inter_edgelist,
												  layer_vec=layer_vec,intra_directed=intra_directed,
												  inter_directed=inter_directed)

# def _save_ml_graph(intralayer_edges,interlayer_edges,layer_vec,filename=None):
#	 if filename is None:
#		 file=tempfile.NamedTemporaryFile()
#	 filename=file.name
#
#	 outdict={"interlayer_edges":interlayer_edges,
#			  'intralayer_edges':intralayer_edges,
#			  'layer_vec':layer_vec}
#
#	 with gzip.open(filename,'w') as fh:
#		 pickle.dump(outdict,fh)
#	 return file #returns the filehandle


def _save_ml_graph(slice_layers,interslice_layer):
	"""
	We save the layers of the graph as graphml.gz files here
	:param slice_layers:
	:param interslice_layer:
	:param layer_vec:
	:return:
	"""
	filehandles=[]
	filenames=[]
	#interslice couplings will be last
	for layer in slice_layers+[interslice_layer]: #save each graph in it's own file handle
		fh=tempfile.NamedTemporaryFile(mode='wb',suffix='.graphml.gz')
		layer.write_graphmlz(fh.name)
		filehandles.append(fh)
		filenames.append(fh.name)
	return filehandles,filenames


def _get_sum_internal_edges_from_partobj_list(part_obj_list,weight='weight'):
	A=0
	for part_obj in part_obj_list:
		A+=get_sum_internal_edges(part_obj,weight=weight)
	return A


def _get_sum_expected_edges_from_partobj_list(part_obj_list,weight='weight'):
	P=0
	for part_obj in part_obj_list:
		#This is the case where we have to split the intralayer adjacency into multiple
		#partition objects.
		P += get_expected_edges(part_obj,weight=weight)
	return P


def _get_modularity_from_partobj_list(part_obj_list,resolution=None):
	finmod=0
	for part_obj in part_obj_list:
		if resolution is None:
			finmod+=part_obj.quality()
		else:
			finmod+=part_obj.quality(resolution_parameter=resolution)
	return finmod

def run_louvain_multilayer(intralayer_graph,interlayer_graph, layer_vec, weight='weight',
						   resolution=1.0, omega=1.0,nruns=1):
	logging.debug('Shuffling node ids')
	t=time()
	mu=np.sum(intralayer_graph.es[weight])+interlayer_graph.ecount()

	use_RBCweighted = hasattr(louvain, 'RBConfigurationVertexPartitionWeightedLayers')

	outparts=[]
	for run in range(nruns):
		rand_perm = list(np.random.permutation(interlayer_graph.vcount()))
		# rand_perm = list(range(interlayer_graph.vcount()))
		rperm = rev_perm(rand_perm)
		interslice_layer_rand = interlayer_graph.permute_vertices(rand_perm)
		rlayer_vec=permute_vector(rand_perm,layer_vec)

		rintralayer_graph=intralayer_graph.permute_vertices(rand_perm)
		#
		if use_RBCweighted:
			rlayers = [intralayer_graph]  #  one layer representing all intralayer connections here
		else:
			rlayers = _create_multilayer_igraphs_from_super_adj_igraph(rintralayer_graph, layer_vec=rlayer_vec)


		logging.debug('time: {:.4f}'.format(time()-t))

		t=time()

		#create the partition objects
		layer_partition_objs=[]

		logging.debug('creating partition objects')
		t=time()

		for i,layer in enumerate(rlayers): #these are the shuffled igraph slice objects
			try:
				res=resolution[i]
			except:
				res=resolution

			if use_RBCweighted:

				cpart=louvain.RBConfigurationVertexPartitionWeightedLayers(layer,layer_vec=rlayer_vec,weights=weight,resolution_parameter=res)
			else:
				#This creates individual VertexPartition for each layer.  Much slower to optimize.
				cpart=louvain.RBConfigurationVertexPartition(layer,weights=weight,resolution_parameter=res)

			layer_partition_objs.append(cpart)

		coupling_partition=louvain.RBConfigurationVertexPartition(interslice_layer_rand,
																  weights=weight,resolution_parameter=0)

		all_layer_partobjs=layer_partition_objs+[coupling_partition]

		optimiser=louvain.Optimiser()
		logging.debug('time: {:.4f}'.format(time()-t))
		logging.debug('running optimiser')
		t=time()


		layer_weights=[1]*len(rlayers)+[omega]
		improvement=optimiser.optimise_partition_multiplex(all_layer_partobjs,layer_weights=layer_weights)

		#the membership for each of the partitions is tied together.
		finalpartition=permute_vector(rperm, all_layer_partobjs[0].membership)
		reversed_partobj=[]
		#go back and reverse the graphs associated with each of the partobj.  this allows for properly calculating exp edges with partobj
		#This is not ideal.  Could we just reverse the permutation?
		for layer in layer_partition_objs:
			if use_RBCweighted:
				reversed_partobj.append(louvain.RBConfigurationVertexPartitionWeightedLayers(graph=layer.graph.permute_vertices(rperm),initial_membership=finalpartition,weights=weight,layer_vec=layer_vec,resolution_parameter=layer.resolution_parameter))
			else:
				reversed_partobj.append(louvain.RBConfigurationVertexPartition(graph=layer.graph.permute_vertices(rperm),initial_membership=finalpartition,weights=weight,resolution_parameter=layer.resolution_parameter))
		coupling_partition_rev=louvain.RBConfigurationVertexPartition(graph=coupling_partition.graph.permute_vertices(rperm),initial_membership=finalpartition,weights=weight,resolution_parameter=0)
		#use only the intralayer part objs
		A=_get_sum_internal_edges_from_partobj_list(reversed_partobj,weight=weight)
		if use_RBCweighted: #should only one partobj here representing all layers
			P= get_expected_edges_ml(reversed_partobj[0], layer_vec=layer_vec, weight=weight)
		else:
			P=_get_sum_expected_edges_from_partobj_list(reversed_partobj,weight=weight)
		C=get_sum_internal_edges(coupling_partition_rev,weight=weight)
		outparts.append({'partition': np.array(finalpartition),
						 'resolution': resolution,
						 'coupling':omega,
						 'orig_mod': (.5/mu)*(_get_modularity_from_partobj_list(reversed_partobj)\
											  +omega*coupling_partition_rev.quality()),
						 'int_edges': A,
						 'exp_edges': P,
						'int_inter_edges':C})

	logging.debug('time: {:.4f}'.format(time()-t))
	return outparts


def _parallel_run_louvain_multimodularity(files_layervec_gamma_omega):
	logging.debug('running parallel')
	t=time()
	# graph_file_names,layer_vec,gamma,omega=files_layervec_gamma_omega
	np.random.seed() #reset seed in forked process
	# louvain.set_rng_seed(np.random.randint(2147483647)) #max value for unsigned long
	intralayer_graph,interlayer_graph,layer_vec,gamma,omega=files_layervec_gamma_omega

	partition=run_louvain_multilayer(intralayer_graph,interlayer_graph, layer_vec=layer_vec, resolution=gamma, omega=omega)
	logging.debug('time: {:.4f}'.format(time()-t))
	return partition


def parallel_multilayer_louvain(intralayer_edges,interlayer_edges,layer_vec,
								gamma_range,ngamma,omega_range,nomega,maxpt=None,numprocesses=2,progress=True,
								intra_directed=False,inter_directed=False):

	"""

	:param intralayer_edges:
	:param interlayer_edges:
	:param layer_vec:
	:param gamma_range:
	:param ngamma:
	:param omega_range:
	:param nomega:
	:param maxpt:
	:param numprocesses:
	:param progress:
	:param intra_directed:
	:param inter_directed:
	:return:

	"""



	logging.debug('creating graphs from edges')
	t=time()
	intralayer_graph,interlayer_graph=create_multilayer_igraph_from_edgelist(
        intralayer_edges=intralayer_edges,
	    interlayer_edges=interlayer_edges,
		layer_vec=layer_vec,inter_directed=inter_directed,
		intra_directed=intra_directed)

	if not hasattr(louvain, 'RBConfigurationVertexPartitionWeightedLayers'):
		warnings.warn(
			"RBConfigurationVertexPartitionWeightedLayers not present in louvain package.  Falling back on creating igraph for each layer.  Note for networks with many layers this can result in considerable slowdown.")


	logging.debug('time {:.4f}'.format(time() - t))
	# logging.debug('graph to file')
	# t = time()
	# fhandles, fnames = _save_ml_graph(slice_layers=[intralayer_graph],
	#								   interslice_layer=interlayer_graph)
	# logging.debug('time {:.4f}'.format(time() - t))

	gammas=np.linspace(gamma_range[0],gamma_range[1],num=ngamma)
	omegas=np.linspace(omega_range[0],omega_range[1],num=nomega)


	args = itertools.product([intralayer_graph],[interlayer_graph], [layer_vec],
							 gammas,omegas)
	tot=ngamma*nomega
	with terminating(Pool(numprocesses)) as pool:
		parts_list_of_list = []
		if progress:
			with tqdm.tqdm(total=tot) as pbar:
				# parts_list_of_list=pool.imap(_parallel_run_louvain_multimodularity,args)
				for i,res in tqdm.tqdm(enumerate(pool.imap(_parallel_run_louvain_multimodularity,args)),miniters=tot):
					# if i % 100==0:
					pbar.update()
					parts_list_of_list.append(res)
		else:
			for i, res in enumerate(pool.imap(_parallel_run_louvain_multimodularity, args)):
				parts_list_of_list.append(res)

	# parts_list_of_list=list(map(_parallel_run_louvain_multimodularity,args)) #testing without parallel.



	all_part_dicts=[pt for partrun in parts_list_of_list for pt in partrun]
	outensemble=PartitionEnsemble(graph=intralayer_graph,interlayer_graph=interlayer_graph,
								  layer_vec=layer_vec,
								  listofparts=all_part_dicts,maxpt=maxpt)

	return outensemble

def parallel_multilayer_louvain_from_adj(intralayer_adj,interlayer_adj,layer_vec,
								gamma_range,ngamma,omega_range,nomega,maxpt=None,numprocesses=2,progress=True,
								intra_directed=False, inter_directed=False):

	"""Call parallel multilayer louvain with adjacency matrices """
	intralayer_edges=adjacency_to_edges(intralayer_adj)
	interlayer_edges=adjacency_to_edges(interlayer_adj)

	return parallel_multilayer_louvain(intralayer_edges=intralayer_edges,interlayer_edges=interlayer_edges,
									   layer_vec=layer_vec,numprocesses=numprocesses,ngamma=ngamma,nomega=nomega,
									   gamma_range=gamma_range,omega_range=omega_range,progress=progress,maxpt=maxpt,
									   intra_directed=intra_directed,inter_directed=inter_directed)

def main():
	return

if __name__=="__main__":
	 sys.exit(main())
