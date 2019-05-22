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
import leidenalg
import numpy as np
import tqdm
from time import time
import warnings
import logging
#logging.basicConfig(format=':%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)
logging.basicConfig(format=':%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

from .louvain_ext import permute_memvec,permute_vector,rev_perm
from .louvain_ext import terminating

iswin = os.name == 'nt'
is_py3 = sys.version_info >= (3, 0)

def run_leiden(gfile,gamma,nruns,weight=None,node_subset=None,attribute=None,niterations=5,
			   output_dictionary=False):
	'''
	Call the leiden method for a given graph file.

	This takes as input a graph file (instead of the graph object) to avoid duplicating
	references in the context of parallelization.  To allow for flexibility, it allows for
	subsetting of the nodes each time.

	:param gfile: igraph file.  Must be GraphMlz
	:param node_subset: Subeset of nodes to keep (either the indices or list of attributes)
	:param gamma: resolution parameter to run leiden
	:param nruns: number of runs to conduct
	:param weight: optional name of weight attribute for the edges if network is weighted.
	:param output_dictionary: Boolean - output a dictionary representation without attached graph.
	:param niterations: int - number of times to iterate leiden for single run.  Input of each iteration\
	is the output of previous iteration.
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



		rp = leidenalg.find_partition(gr,leidenalg.RBConfigurationVertexPartition,weights=weight,
									resolution_parameter=gamma,n_iterations=niterations)

		#old way of calling
		# rp = leiden.find_partition(gr, method='RBConfiguration',weight=weight,  resolution_parameter=gamma)

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


def _run_leiden_parallel(gfile_gamma_nruns_weight_subset_attribute_niters):
	'''
	Parallel wrapper with single argument input for calling :meth:`leiden_ext.run_leiden`
	Unpacks tuple argument and calls run_leiden with arguments.
	:param gfile_att_2_id_dict_shared_gamma_runs_weight: tuple or list of arguments to supply
	:returns: PartitionEnsemble of graph stored in gfile
	'''
	#unpack
	gfile,gamma,nruns,weight,node_subset,attribute,niterations=gfile_gamma_nruns_weight_subset_attribute_niters
	t=time()
	outparts=run_leiden(gfile,gamma,nruns=nruns,weight=weight,node_subset=node_subset,
						attribute=attribute,output_dictionary=True,niterations=niterations)

	return outparts

def parallel_leiden(graph,start=0,fin=1,numruns=200,maxpt=None,niterations=5,
					 numprocesses=None, attribute=None,weight=None,node_subset=None,progress=None):
	'''
	Generates arguments for parallel function call of leiden on graph

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
			cpart_ens=run_leiden_windows(graph=graph,nruns=1,gamma=gam,node_subset=node_subset,
										attribute=attribute,weight=weight,niterations=niterations)
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
		parallel_args.append((graphfile, curg, 1, weight, None, None,niterations))

	parts_list_of_list=[]
	# use a context manager so pools properly shut down
	with terminating(Pool(processes=numprocesses)) as pool:

		if progress:
			tot = len(parallel_args)
			with tqdm.tqdm(total=tot) as pbar:
				# parts_list_of_list=pool.imap(_parallel_run_leiden_multimodularity,args)
				for i, res in tqdm.tqdm(enumerate(pool.imap(_run_leiden_parallel, parallel_args)), miniters=tot):
					# if i % 100==0:
					pbar.update()
					parts_list_of_list.append(res)
		else:
			parts_list_of_list=pool.map(_run_leiden_parallel, parallel_args )

	#for debugging
	# parts_list_of_list=list(map(_run_leiden_parallel,parallel_args))

	all_part_dicts=[pt for partrun in parts_list_of_list for pt in partrun]
	tempf.close()
	outensemble=PartitionEnsemble(graph,listofparts=all_part_dicts,maxpt=maxpt)
	return outensemble


def run_leiden_windows(graph,gamma,nruns,weight=None,node_subset=None,attribute=None,output_dictionary=False,niterations=5):
	'''
	Windows version takes the input graph directly (not file).

	:param graph: igraph
	:param node_subset: Subeset of nodes to keep (either the indices or list of attributes)
	:param gamma: resolution parameter to run leiden
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
		rp = leidenalg.find_partition(gr,leidenalg.RBConfigurationVertexPartition,
									n_iterations=niterations,weights=weight,
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