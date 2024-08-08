# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""multi_view_clustering.py: Contains methods (in classes) for clustrering
multi-view, social-based data."""

__author__  = "Iain Cruickshank"
__license__ = "MIT"
__version__ = "0.2"
__email__   = "icruicks@andrew.cmu.edu"

import numpy as np, pandas as pd, igraph as ig, leidenalg as la
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, issparse
from sklearn.base import ClusterMixin
from sknetwork.clustering import Louvain, get_modularity
from sklearn.utils.validation import check_symmetric


class CVIC(ClusterMixin):
    """Cross-View Influence Clustering for clustering multi-view data
    
    Cross-View Influence Clustering (CVIC) is a hybrid paradigm clustering
    technique for clustering multi-view data. It uses graphs of each view
    and a set of cluster labels from each. Note, the order of vertices in the
    graphs and cluste labels must all match.
    
    Parameters
    ----------
    modal_graphs : list of array_like or sparse matrix
        list of len number of views of scipy sparse matrices or other arrays
        of shape n-by-n.
    base_clusters : array_like
        cluster labels from each view of size n-by-number of views
    alpha : float
        graph influence term; controls how mush influence is applied by
        neighbors in the view graphs on each objects clustering labels. default
        is 0.9
    iterations : int
        number of iterations to run the cross influence procedure before doing
        the final clustering
    
    Returns
    -------
    z : ndarray
        result of shape (n,).
    """
    
    def __init__(self, alpha=0.9, iterations=10):
        
        self.iterations = iterations
        self.alpha = alpha
        self.W = []
    
    def fit_predict(self, base_clusters, modal_graphs):
        ba_matrices =[]
        idxs = []
        idxs_iter = 0
        
        if isinstance(clusters, pd.DataFrame):
            clusters = [clusters.loc[:,i].to_numpy(dtype=int) for i in clusters.columns]
        else:
            clusters = [clustering.astype(np.int64) for clustering in clusters]
        
        for base_cluster in base_clusters:
            ba_matrix = np.zeros((base_cluster.size, base_cluster.max()+1))
            ba_matrix[np.arange(base_cluster.size), base_cluster] = 1
            ba_matrices.append(ba_matrix)
            idxs.append([idxs_iter, idxs_iter+ba_matrix.shape[1]])
            idxs_iter += ba_matrix.shape[1]
            
        ba_matrix = np.concatenate(ba_matrices, axis=1)
        
        
        for G in modal_graphs:
            self.W.append(normalize(csr_matrix(G).T, norm='l1', axis=1, copy=True))
            
        ca_matrix = ba_matrix.copy()

        for t in range(self.iterations):
            temp = []
            
            for G_m in self.W:
                temp.append(self.alpha * G_m @ ca_matrix + (1-self.alpha)* ba_matrix )
                
            ca_matrix = np.mean(temp, axis=0)

        self._ca_matrix = ca_matrix
        clstr = Louvain()
        clstr.fit(self._ca_matrix)
        self._object_labels = clstr.labels_row_
        self._cluster_labels = clstr.labels_col_
        self._quality = get_modularity(self._ca_matrix, self._object_labels, 
                                      self._cluster_labels)
        return self._object_labels



class MVMC(ClusterMixin):
    """Multi-view Modualrity Clustering for clustering multi-view data
    
    Multi-view Modualrity Clustering (MVMC) is an intermediate paradigm clustering
    technique for clustering multi-view data. It uses graphs of each view. Note, 
    the order of vertices in the graphs must match across all views.
    
    Parameters
    ----------
    graphs : list of array_like or sparse matrix
        list of len number of views of scipy sparse matrices or other arrays
        of shape n-by-n.
    n_iterations : int
        number of iterations of Leiden algorithm to use in each clustering step.
        defualt is -1, which means run until modualrity does not improve
    max_clusterings : int
        maximum number of iterations to run the method. default is 20
    resolution_tol : float
        tolerance for determining convergence based on the resolution values not
        changing
    weight_tol : float
        tolerance for determining convergence based on the weight values not
        changing
    verbose : boolean
        wether to print intermediate step information. Can be useful
        in diagnosing clustering issues and understanding data
    
    Returns
    -------
    z : ndarray
        result of shape (n,).
    """
    
    def __init__(self, n_iterations=-1, max_clusterings=20, 
                 resolution_tol=1e-2, weight_tol=1e-2, verbose=False):
        
        self.n_iterations = n_iterations
        self.max_clusterings = max_clusterings
        self.resolution_tol = resolution_tol
        self.weight_tol = weight_tol
        self.verbose = verbose
        
    def fit_transform(self, graphs):
        G=[]
        for graph in graphs:
            if type(graph) is ig.Graph:
                G.append(graph)
            elif issparse(graph):
                G.append(self._scipy_to_igraph(graph))
            else:
                G.append(self._other_to_igraph(graph))
                
        if self.verbose:
            for i in range(len(G)):
                print("View Graph {}: num_nodes: {}, num_edges: {}, directed: {}, num_components: {}, num_isolates: {}"
                      .format(i, G[i].vcount(), G[i].ecount(), G[i].is_directed(), 
                              len(G[i].components(mode='WEAK').sizes()), G[i].components(mode='WEAK').sizes().count(1)))
        
        self.weights = []
        self.resolutions =[]
        self.best_modularity =-np.inf
        self.best_clustering = None
        self.best_resolutions = None
        self.best_weights = None
        self.modularities =[]
        self.clusterings =[]
        self.final_iteration = 0
        self.best_iteration = 0
        
        weights = [1]*len(G)
        resolutions =[1]*len(G)
        
        for iterate in range(self.max_clusterings):
            partitions = []
            for i in range(len(G)):
                partitions.append(la.RBConfigurationVertexPartition(G[i], resolution_parameter=resolutions[i]))
                
            optimiser = la.Optimiser()
            diff = optimiser.optimise_partition_multiplex(partitions, layer_weights = weights, n_iterations=self.n_iterations)
            self.clusterings.append(np.array(partitions[0].membership))
            self.modularities.append([part.quality()/(part.graph.ecount() if part.graph.is_directed() else 2*part.graph.ecount()) 
                                      for part in partitions])
            self.weights.append(weights.copy())
            self.resolutions.append(resolutions.copy())
            self.final_iteration +=1
            
            
            if self.verbose:
                print("--------")
                print("Iteration: {} \n Modularities: {} \n Resolutions: {} \n Weights: {}"
                      .format(self.final_iteration, self.modularities[-1], resolutions, weights))
            
            if np.sum(np.array(self.weights[-1]) * np.array(self.modularities[-1])) > self.best_modularity:
                self.best_clustering = self.clusterings[-1]
                self.best_modularity = np.sum(np.array(self.weights[-1]) * np.array(self.modularities[-1]))
                self.best_resolutions = self.resolutions[-1]
                self.best_weights = self.weights[-1]
                self.best_iteration = self.final_iteration
            
            theta_in, theta_out = self._calculate_edge_probabilities(G)
            for i in range(len(G)):
                resolutions[i] = (theta_in[i] - theta_out[i])/ (np.log(theta_in[i]) - np.log(theta_out[i]))
                weights[i] = (np.log(theta_in[i]) - np.log(theta_out[i]))/(np.mean([np.log(theta_in[j]) - np.log(theta_out[j]) for j in range(len(G))]))

                
            if (np.all(np.abs(np.array(self.resolutions[-1])-np.array(resolutions)) <= self.resolution_tol)
                and np.all(np.abs(np.array(self.weights[-1])-np.array(weights)) <= self.resolution_tol)):
                break
        else:
            best_iteration = np.argmax([np.sum(np.array(self.weights[i]) * np.array(self.modularities[i]))
                                        for i in range(len(self.modularities))])
            self.best_clustering = self.clusterings[best_iteration]
            self.best_modularity = np.sum(np.array(self.weights[best_iteration]) * np.array(self.modularities[best_iteration]))
            self.best_resolutions = self.resolutions[best_iteration]
            self.best_weights = self.weights[best_iteration]
            self.best_iteration = best_iteration
            
            if self.verbose:
                print("MVMC did not converge, best result found: Iteration: {}, Modularity: {}, Resolutions: {}, Weights: {}"
                      .format(self.best_iteration, self.best_modularity, self.best_resolutions, self.best_weights))


        return self.best_clustering
    
    
    def _scipy_to_igraph(self, matrix):
        matrix.eliminate_zeros()
        sources, targets = matrix.nonzero()
        weights = matrix.data
        graph = ig.Graph(n=matrix.shape[0], edges=list(zip(sources, targets)), directed=True, edge_attrs={'weight': weights})
        #graph = ig.Graph.Weighted_Adjacency(matrix, attr="weight")
        
        try:
            check_symmetric(matrix, raise_exception=True)
            graph = graph.as_undirected(combine_edges="mean")
        except ValueError:
            pass
        
        return graph
    
    
    def _other_to_igraph(self, matrix):
        if isinstance(matrix, pd.DataFrame):
            A = matrix.values
        else:
            A= matrix
        
        graph = ig.Graph.Adjacency((A > 0).tolist())
        graph.es['weight'] = A[A.nonzero()]
        
        return graph

    
    def _calculate_edge_probabilities(self, G):
        theta_in =[]
        theta_out =[]
        clusters = self.clusterings[-1].copy()
        for i in range(len(G)):
            m_in = 0
            m = sum(e['weight'] for e in G[i].es)
            kappa =[]
            G[i].vs['clusters'] = clusters
            for cluster in np.unique(clusters):
                nodes = G[i].vs.select(clusters_eq=cluster)
                m_in += sum(e['weight'] for e in G[i].subgraph(nodes).es)
                if G[i].is_directed():
                    degree_products = np.outer(np.array(G[i].strength(nodes, mode = 'IN', weights='weight')), 
                                               np.array(G[i].strength(nodes, mode = 'OUT', weights='weight')))
                    np.fill_diagonal(degree_products,0)
                    kappa.append(np.sum(degree_products, dtype=np.int64))
                else:
                    kappa.append(np.sum(np.array(G[i].strength(nodes, weights='weight')), dtype=np.int64)**2)
            
            if G[i].is_directed():
                if m_in <=0:
                    # Case when there are no internal edges; every node in its own  cluster
                    theta_in.append(1/G[i].ecount())
                else:
                    theta_in.append((m_in)/(np.sum(kappa, dtype=np.int64)/(2*m)))
                if m-m_in <=0:
                    # Case when all edges are internal; 1 cluster or a bunch of disconnected clusters
                    theta_out.append(1/G[i].ecount())
                else:
                    theta_out.append((m-m_in)/(m-np.sum(kappa, dtype=np.int64)/(2*m)))
            else:
                if m_in <=0:
                    # Case when there are no internal edges; every node in its own  cluster
                    theta_in.append(1/G[i].ecount())
                else:
                    theta_in.append((m_in)/(np.sum(kappa, dtype=np.int64)/(4*m)))
                if m-m_in <=0:
                    # Case when all edges are internal; 1 cluster or a bunch of disconnected clusters
                    theta_out.append(1/G[i].ecount())
                else:
                    theta_out.append((m-m_in)/(m-np.sum(kappa, dtype=np.int64)/(4*m)))

        return theta_in, theta_out
        
        
        
        
        
        
        
    