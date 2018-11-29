from context import champ
import igraph as ig
import numpy as np
import scipy.io as scio
import pandas as pd
import sys, os
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
from time import time
import modbp
import forceatlas2 as fa2
import seaborn as sbn
import sklearn.metrics as skm
import champ
import louvain
import scipy.io as scio
import matplotlib.pyplot as plt


def test_parallel_run():
	G = ig.Graph.Erdos_Renyi(n=100, p=.04, directed=False)

	G.es['weight']=np.random.normal(loc=10,scale=1,size=G.ecount())
	part_ens = champ.parallel_louvain(G,
												start=.1,
												fin=5.1,
												numruns=10,
												weight='weight',
												progress=True)

	inds=part_ens.get_CHAMP_indices()

	partition=part_ens.partitions[inds[0]]
	gamma=part_ens.ind2doms[inds[0]][0][0]
	A=part_ens.int_edges[inds[0]]
	P=part_ens.exp_edges[inds[0]]
	print("champ mod: ",A-gamma*P)
	print(part_ens.orig_mods[0])

	print()

	plt.close()
	part_ens.plot_modularity_mapping()
	plt.show()
	print (part_ens)


def test_time_multilayer():


	total_n=[1000,2000,4000,5000,10000]
	nlayers_vals=[10,20,50]
	n = 100
	q = 4
	nblocks = q
	c = 4
	ep = .1
	eta = .1
	pin = (n * c / (2.0)) / ((n / float(q)) * (n / float(q) - 1) / 2.0 * float(q) + ep * (q * (q - 1) / 2.0) * (
		np.power(n / (q * 1.0), 2.0)))

	pout = ep * pin
	prob_mat = np.identity(nblocks) * pin + (np.ones((nblocks, nblocks)) - np.identity(nblocks)) * pout

	times=pd.DataFrame(columns=['method','n','nlayers','totaln','time'])

	for totn in total_n:

		for nlayers in nlayers_vals:
			n=totn/nlayers
			print(n,nlayers)
		
			ml_sbm = modbp.MultilayerSBM(n, comm_prob_mat=prob_mat, layers=nlayers, transition_prob=eta)
			mgraph = modbp.MultilayerGraph(intralayer_edges=ml_sbm.intraedges, interlayer_edges=ml_sbm.interedges,
										   layer_vec=ml_sbm.layer_vec,
										   comm_vec=ml_sbm.get_all_layers_block())

			layers, interslice, g_full = champ.call_slices_to_layers_from_edge_list(intralayer_edges=mgraph.intralayer_edges,
																					interlayer_edges=mgraph.interlayer_edges,
																					layer_vec=mgraph.layer_vec)
			gamma=1.0
			omega=1
			args = (layers, interslice, gamma, omega)
			t=time()
			part = champ.louvain_ext._parallel_run_louvain_multimodularity(args)
			t=time()-t
			cind=times.shape[0]
			times.loc[cind,['method','n','nlayers','totaln','time']]='old',n,nlayers,totn,t

			#run the new version
			t=time()

			intraslice, interslice = champ.create_multilayer_igraph_from_edgelist(intralayer_edges=mgraph.intralayer_edges,
																				  interlayer_edges=mgraph.interlayer_edges,
																				  layer_vec=mgraph.layer_vec)

			RBCpartobj = louvain.RBConfigurationVertexPartitionWeightedLayers(intraslice, resolution_parameter=1.0,
																			  layer_vec=mgraph.layer_vec.tolist())
			InterlayerPartobj = louvain.RBConfigurationVertexPartition(interslice, resolution_parameter=0.0)

			opt = louvain.Optimiser()
			opt.optimise_partition_multiplex(partitions=[RBCpartobj, InterlayerPartobj])
			t = time() - t
			cind = times.shape[0]
			times.loc[cind, ['method','n', 'nlayers', 'totaln', 'time']] ='new',n, nlayers, totn, t


	plt.close()
	# f,a=plt.subplots(1,1,figsize=(10,5))
	sbn.lmplot(x='totaln',y='time',hue='nlayers',data=times,col='method',fit_reg=False)
	plt.suptitle("Runtime Louvain on multilayer SBM")
	plt.savefig('louvain_runtimes_multilayer_comparison.pdf')
	plt.show()


	# print(skm.adjusted_mutual_info_score(part[0]['partition'],mgraph.comm_vec))

	# plt.close()
	# a=plt.subplot2grid((1,2),(0,0))
	# a.set_title('Ground')
	# champ.plot_multilayer_community(mgraph.comm_vec,mgraph.layer_vec,ax=a)
	# a=plt.subplot2grid((1,2),(0,1))
	# a.set_title("Multilayer gamma={:.3f},omega={:.3f}".format(gamma,omega))
	# champ.plot_multilayer_community(part[0]['partition'],mgraph.layer_vec,ax=a)
	# plt.gcf().set_size_inches((10,5))
	# plt.show()

def test_multilayer_louvain():
	n = 200
	q = 2
	nlayers = 5
	nblocks = q
	c = 8
	ep = .1
	eta = .1
	pin = (n * c / (2.0)) / ((n / float(q)) * (n / float(q) - 1) / 2.0 * float(q) + ep * (q * (q - 1) / 2.0) * (
		np.power(n / (q * 1.0), 2.0)))

	pout = ep * pin

	prob_mat = np.identity(nblocks) * pin + (np.ones((nblocks, nblocks)) - np.identity(nblocks)) * pout

	ml_sbm = modbp.MultilayerSBM(n, comm_prob_mat=prob_mat, layers=nlayers, transition_prob=eta)
	mgraph = modbp.MultilayerGraph(intralayer_edges=ml_sbm.intraedges, interlayer_edges=ml_sbm.interedges,
								   layer_vec=ml_sbm.layer_vec,
								   comm_vec=ml_sbm.get_all_layers_block())

	intralayer, interlayer = champ.create_multilayer_igraph_from_edgelist(intralayer_edges=mgraph.intralayer_edges,interlayer_edges=mgraph.interlayer_edges,layer_vec=mgraph.layer_vec)

	#run a sinlge time and compare with ground truth
	np.random.seed(0)
	# gamma = 1.0
	# omega = 2
	# args = (intralayer,interlayer,mgraph.layer_vec, gamma, omega)
	# part = champ.louvain_ext._parallel_run_louvain_multimodularity(args)
	# print('modularity',part[0]['orig_mod'])
	# print(skm.adjusted_mutual_info_score(mgraph.comm_vec,part[0]['partition']))
	# # test parallel version
	#
	# # ML_PartEnsemble=champ.louvain_ext.parallel_multilayer_louvain(intralayer_edges=mgraph.intralayer_edges,interlayer_edges=mgraph.interlayer_edges,layer_vec=mgraph.layer_vec,gamma_range=[1,1],ngamma=1,omega_range=[1,1],nomega=1,numprocesses=1,maxpt=(1,1))
	#
	# plt.close()
	# f,a=plt.subplots(1,2,figsize=(10,5))
	# a=plt.subplot(1,2,1)
	# champ.plot_multiplex_community(mgraph.comm_vec, mgraph.layer_vec,ax=a)
	# a=plt.subplot(1,2,2)
	# champ.plot_multiplex_community(part[0]['partition'], mgraph.layer_vec,ax=a)
	# # ML_PartEnsemble.plot_multiplex_communities(ind=0,ax=a)
	# plt.show()


	#run several

	ML_PartEnsemble=champ.louvain_ext.parallel_multilayer_louvain(intralayer_edges=mgraph.intralayer_edges,
																  interlayer_edges=mgraph.interlayer_edges,
																  layer_vec=mgraph.layer_vec,gamma_range=[0,4],
																  ngamma=2,omega_range=[0,2],nomega=2,numprocesses=2,maxpt=(4,2),progress=True)



	print("Size of CHAMP: {:d} of {:d} runs".format(len(ML_PartEnsemble.ind2doms),ML_PartEnsemble.numparts))
	print('number unique parititons : {:d}'.format(len(ML_PartEnsemble.get_unique_coeff_indices())))

	filename=ML_PartEnsemble.save()
	print(filename)
	# ML_PartEnsemble2=champ.PartitionEnsemble().open(filename)
	# ML_PartEnsemble2=ML_PartEnsemble.merge_ensemble(ML_PartEnsemble2,new=True)
	# ML_PartEnsemble2.save(filename='test2_partensemble.hdf5')

	# print("Size of CHAMP: {:d} of {:d} runs".format(len(ML_PartEnsemble2.ind2doms),ML_PartEnsemble2.numparts))
	# print('number unique parititons : {:d}'.format(len(ML_PartEnsemble2.get_unique_coeff_indices())))


	print()
	plt.close()
	# f,a=plt.subplots(1,2,figsize=(14,7))
	a=plt.subplot2grid((1,11),(0,0),colspan=5)
	a=ML_PartEnsemble.plot_2d_modularity_domains(ax=a)
	print (ML_PartEnsemble.ind2doms.keys())
	a=plt.subplot2grid((1,11),(0,5),colspan=5)
	a_cb=plt.subplot2grid((1,11),(0,10),colspan=1)
	# ML_PartEnsemble.apply_CHAMP(subset=list(ML_PartEnsemble.ind2doms))

	#rescale and see if domains are affected
	# mu=ML_PartEnsemble.mu
	# ML_PartEnsemble.int_inter_edges /= (2.0*mu)
	# ML_PartEnsemble.int_edges /= (2*mu)
	# ML_PartEnsemble.exp_edges /= (2*mu)
	# ML_PartEnsemble.apply_CHAMP()
	# print(ML_PartEnsemble.ind2doms.keys())
	# print("Size of CHAMP : {:d} of {:d} runs".format(len(ML_PartEnsemble.ind2doms),ML_PartEnsemble.numparts))
	a=ML_PartEnsemble.plot_2d_modularity_domains_with_AMI(ax=a,true_part=mgraph.comm_vec,colorbar_ax=a_cb)
	plt.show()


def get_layer_average_ami(labs1,labs2,layer_vec):
	layers,cnts=np.unique(layer_vec,return_counts=True)
	allamis=[]
	for i,layer in enumerate(layers):
		cinds=np.where(layer_vec==layer)[0]
		cami=skm.adjusted_mutual_info_score(labs1[cinds],labs2[cinds])
		cami=cami*cnts[i]/(1.0*len(layer_vec))
		allamis.append(cami)
	return np.sum(allamis)

def test_create_coeff_array():
	n = 200
	q = 2
	nlayers = 3
	nblocks = q
	c = 8
	ep = .1
	eta = .1
	pin = (n * c / (2.0)) / ((n / float(q)) * (n / float(q) - 1) / 2.0 * float(q) + ep * (q * (q - 1) / 2.0) * (
		np.power(n / (q * 1.0), 2.0)))

	pout = ep * pin

	prob_mat = np.identity(nblocks) * pin + (np.ones((nblocks, nblocks)) - np.identity(nblocks)) * pout

	ml_sbm = modbp.MultilayerSBM(n, comm_prob_mat=prob_mat, layers=nlayers, transition_prob=eta)
	mgraph = modbp.MultilayerGraph(intralayer_edges=ml_sbm.intraedges, interlayer_edges=ml_sbm.interedges,
								   layer_vec=ml_sbm.layer_vec,
								   comm_vec=ml_sbm.get_all_layers_block())

	intralayer, interlayer = champ.create_multilayer_igraph_from_edgelist(intralayer_edges=mgraph.intralayer_edges
																		  ,
																		  interlayer_edges=mgraph.interlayer_edges,
																		  layer_vec=mgraph.layer_vec)

	P = np.zeros((nlayers * n, nlayers * n))
	for i in range(nlayers):
		c_degrees = np.array(intralayer.degree(list(range(n * i, n * i + n))))
		c_inds = np.where(mgraph.layer_vec == i)[0]
		P[np.ix_(c_inds, c_inds)] = np.outer(c_degrees, c_degrees.T) / (1.0 * np.sum(c_degrees))

	ML_PartEnsemble=champ.louvain_ext.parallel_multilayer_louvain(intralayer_edges=mgraph.intralayer_edges,
																  interlayer_edges=mgraph.interlayer_edges,
																  layer_vec=mgraph.layer_vec,gamma_range=[1,5],
																  ngamma=5,omega_range=[1,2],nomega=1,
																  numprocesses=4,maxpt=(12,12))

	man_array=champ.create_coefarray_from_partitions(ML_PartEnsemble.partitions,A_mat=ML_PartEnsemble.get_adjacency(),
													 P_mat=P,C_mat=ML_PartEnsemble.get_adjacency(intra=False))

	print(man_array)
	print(ML_PartEnsemble.get_coefficient_array())

	print(champ.get_intersection(man_array,max_pt=(12,12)).keys())
	print(champ.get_intersection(man_array,max_pt=(12,12)).keys())

	print(ML_PartEnsemble.ind2doms.keys())
	
def test_on_senate_data():

	senate_dir = '/Users/whweir/Documents/UNC_SOM_docs/Mucha_Lab/Mucha_Python/modularity_domains/multilayer_senate'
	senate_data_file = os.path.join(senate_dir, 'multisenate0.5.mat')
	sendata = scio.loadmat(senate_data_file)

	A = sendata['A']
	C = sendata['C']
	sesid = sendata['Ssess'][:, 0]
	parties = sendata['Sparty'][:, 0]
	sessions = np.unique(sesid)
	sess2layer = dict(zip(sessions, range(len(sessions))))
	layer_vec = np.array(list(map(lambda x: sess2layer[x], sesid)))
	intralayer,interlayer=champ.create_multilayer_igraph_from_adjacency(A=A,C=C,layer_vec=layer_vec)

	parts = champ.parallel_multilayer_louvain_from_adj(intralayer_adj=A, interlayer_adj=C,
	                                                   layer_vec=layer_vec, numprocesses=10,
	                                                   gamma_range=[1, 2], omega_range=[0, 2],
	                                                   ngamma=4, nomega=4, maxpt=(2, 2))

	#test single run
	gamma = .8
	omega = 1.0
	args = (intralayer, interlayer,layer_vec, gamma, omega)
	# args=(champ.louvain_ext.adjacency_to_edges(A),
	#	   champ.louvain_ext.adjacency_to_edges(C),
	#	   layer_vec,1.0,.5)
	part = champ.louvain_ext._parallel_run_louvain_multimodularity(args)
	print(skm.adjusted_mutual_info_score(part[0]['partition'], parties))
	print(get_layer_average_ami(part[0]['partition'],parties,layer_vec))
	print()

	#test parallel function


	# plt.close()
	# a = plt.subplot2grid((1, 2), (0, 0))
	# a.set_title('Parties')
	# champ.plot_multilayer_community(parties, layer_vec, ax=a)
	# a = plt.subplot2grid((1, 2), (0, 1))
	# a.set_title("Multilayer gamma={:.3f},omega={:.3f}".format(gamma, omega))
	# champ.plot_multilayer_community(part[0]['partition'], layer_vec, ax=a)
	# plt.gcf().set_size_inches((10, 5))
	# plt.show()

def main():
	test_parallel_run()
	return 0

if __name__=='__main__':
	main()
