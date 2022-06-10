from ego_utils import get_ego_src, deadend, get_ego_dst, ego_edges_df
from measures import measures
from utils import tab_printer

import random
import pandas as pd
import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigs
from tqdm import tqdm
from os import path
import networkx as nx
import time as time
"""
Todo: 
    1. Node degree sampler assumes that directed graph node feature include the degree value.
    2. write sampling deal with non existance folder for a given dataset. 
    3. Update the induced subgraph function. so for the senario that ego exist it behave differently. 

"""
class Sampler:
    def __init__(self, args, print_ = True):
        self.args = args
        self.print_ = print_  
        self.__load_data()
        #self.__original_graph_statistics()      
        self.measures = measures(args, print_)
        if not(self.args.ego_exist):
            self.__load_data(load_edges = True)

    def __load_data(self, load_edges = False):
        if load_edges:
            #self.__edges = pd.read_csv(self.args.path + 'relations.csv')[['src', 'dst']]
            self.__edges = pd.read_pickle(self.args.path + "relations.pkl")
        else:
            if self.args.undirected ==1:
                #self.__cent = pd.read_csv(self.args.path + 'node_centralities.csv', index_col=0)
                self.__cent = pd.read_pickle(self.args.path+"node_centralities.pkl")
            #self.__users = pd.read_csv(self.args.path + 'usersdata.csv' ).set_index('user_id')
            self.__users = pd.read_pickle(self.args.path+"usersdata.pkl")
        
    def run_sample(self, rand_seed = 0):
        np.random.seed(0)
        if self.args.random_target:
            target = list(np.random.choice(list(self.__users[self.__users[self.args.protected] == 1].index),10)) # choose 10 target nodes from first group. 
            target += list(np.random.choice(list(self.__users[self.__users[self.args.protected] == 0].index),10)) # choose 10 target nodes from second group.
        else:
            target = list(self.__cent.loc[self.__users[self.__users[self.args.protected] == 1].index, 'degree'].nlargest().index) #choose 5 target nodes from first group with highest harmonic value 
            target += list(self.__cent.loc[self.__users[self.__users[self.args.protected] == 0].index, 'degree'].nlargest().index) #choose 5 target nodes from second group with highest harmonic value
            print('high degree targets',target)
#         target = [1310732, 5408896, 99104, 3970275, 3258932, 91281, 3132368, 1622288, 343440, 4443984, 2228262]
        
    
        np.random.seed(rand_seed)
        self.__users[self.__users[self.args.protected] == 1].index
        if self.print_:
            print('>>> Start the sampling process ...')
        if self.args.sampler == 'NS':
            nodes = self.__node_sampler(target)
        elif self.args.sampler == 'NSD':
            nodes = self.__node_sampler_degree(target)
        elif self.args.sampler == 'RW':
            nodes = self.__random_walk(target)            
        elif self.args.sampler == 'FRW':
            nodes = self.__fair_random_walk(target)
        elif self.args.sampler == 'MHRW':
            nodes = self.__MH_random_walk(target)
        elif self.args.sampler == 'GG':
            nodes = self.__greedy(target)
        elif self.args.sampler == 'GF':
            map_node_group = dict()
            for node in target:
                map_node_group[node] = self.__users.loc[node, self.args.protected]
            nodes = self.__greedy(target, map_node_group)        
        elif self.args.sampler == 'DFS':
            nodes = self.__DFS(target)
        elif self.args.sampler == 'BFS':
            nodes = self.__BFS(target)
        return nodes    
        


    def write_sample(self, nodes , file_id):
        if self.print_:
            print('>>> Start writing process...')
        file_name = self.__output_name(file_id)

        print(file_name)
        if self.args.random_target:
            file_ = open('random/' +file_name,'w')
        else:
            file_ = open('degree/' +file_name,'w')
        line = str()
        for node in nodes:
            line += str(node)+ ' '
        line += '\n'
        file_.write(line)
        file_.close()
    
    def __output_name(self, file_id ,sampler = None):
        name = self.args.path.split('./data/')[0]
        if sampler is None:
            name += self.args.sampler + '_' + str(self.args.sample_size)
        else:
            name += sampler + '_' + str(self.args.sample_size)
        name += '_'+ str(file_id) +'.txt'
        return name  
    
        
    
    def __ego_network(self, node,  task ):
        if self.args.ego_exist: 
            if task == 'ego_src':
                return get_ego_src(node,self.args.path)
            elif task == 'deadend':
                return deadend(node, self.args.path)
            elif task == 'ego_dst':
                return get_ego_dst(node,self.args.path)
        else:
            if task == 'ego_src':
                if self.args.undirected ==1:
                    return list(self.__edges[self.__edges.dst == node].src) + list(self.__edges[self.__edges.src == node].dst)
                else:
                    return list(self.__edges[self.__edges.src == node].dst)
            elif task == 'deadend':
                dsts = self.__edges [self.__edges.src == node].dst.unique()
                if len(dsts) == 0:
                    return True
                elif (len(dsts) > 1) or (dsts[0] != node): 
                    return False
                return True
            elif task == 'ego_dst':
                return list(self.__edges[self.__edges.dst == node].src)


    
    def __node_sampler(self,target):
        population = list(set(self.__cent.index) - set(target))
        return target+list(np.random.choice(population, self.args.sample_size - len(target), replace = False))
        
    def __node_sampler_degree(self, target):
        #users = pd.read_csv(self.args.feat_path, index_col=0 )
        #self.__load_data()
        if self.args.undirected ==1:
            p = self.__cent.degree.to_numpy()
            p = p/np.sum(p)
        else:
            in_deg = self.__users.in_degree.to_numpy()
            out_deg = self.__users.out_degree.to_numpy()
            in_deg = in_deg/np.sum(in_deg)
            out_deg = out_deg/np.sum(out_deg)
            p = 0.5* in_deg + 0.5*out_deg
        sampled = set(np.random.choice(self.__cent.index, self.args.sample_size,p = p ,replace = False))
        sampled = sampled - set(target)
        
        return target + list(sampled)[:self.args.sample_size-len(target)]
        
    def __random_walk(self, target, reset_p = 0.3):
        '''
        generate random walks on graph.
        input:
            args.ego_path: input ego network data directory
            args.feat_path : the node feature file.
            args.sample_size: size of sample
        output:
            nodes = list of sampled nodes


        '''
        # randomly choose a node to start the walk, which is not a leaf(deadend).
        #users = pd.read_csv(self.args.feat_path, index_col=0 )
        #self.__load_data()
        if self.args.undirected ==1:
            population = set(self.__cent.index)
        else:
            population = set(self.__users[self.__users.out_degree > 0].index)
        observed = set()
        sampled = set()
        while len(sampled)< self.args.sample_size:
            node_start = np.random.choice(target)
            node_c = node_start # current node
            observed.add(node_c)
            while len(observed - sampled)> 0:
            #for i in range(self.__users.shape[0]): # main loop of the walk
                sampled.add(node_c)
                neighbors = set(self.__ego_network(node_c, 'ego_src'))
                observed = observed.union(neighbors)
                if np.random.uniform() > reset_p:
                    node_c = np.random.choice(list(neighbors))
                else:
                    node_c = np.random.choice(target)
                #if deadend(node_c, self.args.path): # if the new node is deadend start over from start node.
                if len(sampled)>= self.args.sample_size:
                    break
        return list(sampled)
    
    def __fair_random_walk(self, target, reset_p = 0.3):
        '''
        generate random walks on graph.
        input:
            args.ego_path: input ego network data directory
            args.feat_path : the node feature file.
            args.sample_size: size of sample
        output:
            nodes = list of sampled nodes


        '''
        # randomly choose a node to start the walk, which is not a leaf(deadend).
        #users = pd.read_csv(self.args.feat_path, index_col=0 )
        #self.__load_data()
        if self.args.undirected ==1:
            population = set(self.__cent.index)
        else:
            population = set(self.__users[self.__users.out_degree > 0].index)
        observed = set()
        sampled = set()
        while len(sampled)< self.args.sample_size:
            node_start = np.random.choice(target)
            node_c = node_start # current node
            observed.add(node_c)
            while len(observed - sampled)> 0:
            #for i in range(self.__users.shape[0]): # main loop of the walk
                sampled.add(node_c)
                neighbors = set(self.__ego_network(node_c, 'ego_src'))
                observed = observed.union(neighbors)
                if np.random.uniform() > reset_p:
                    groups = self.__prt_partition(neighbors)
                    group = np.random.choice(groups)
                    node_c = np.random.choice(list(group))
                else:
                    node_c = np.random.choice(target)
                #if deadend(node_c, self.args.path): # if the new node is deadend start over from start node.
                if len(sampled)>= self.args.sample_size:
                    break
        return list(sampled)
    
    def __prt_partition(self, node_set):
        groups = []
        for val in range(2):
            group = set((self.__users.loc[list(node_set), self.args.protected]==val).index)
            if len(group) > 0:        
                groups .append(group)   
        return groups 

    def __MH_random_walk(self, target):
        '''
        generate Metropolis-Hastings random walks on graph.
        input:
            args.ego_path: input ego network data directory
            args.feat_path : the node feature file.
            args.sample_size: size of sample
        output:
            nodes = list of sampled nodes


        '''
        # randomly choose a node to start the walk, which is not a leaf(deadend).
        #users = pd.read_csv(self.args.feat_path, index_col=0 )
        #self.__load_data()
        if self.args.undirected ==1:
            population = set(self.__users.index)
        else:
            population = set(self.__users[self.__users.out_degree > 0].index)
        observed = set()
        sampled = set()
        while len(sampled)< self.args.sample_size:
            node_start = np.random.choice(list(population-sampled))
            #while deadend(node_start, self.args.path):
            observed.add(node_start)
            sampled.add(node_start)
            neighbors_start = set(self.__ego_network(node_start, 'ego_src'))
            degree_start = float(len(neighbors_start))
            observed = observed.union(neighbors_start)
            node_current = node_start
            neighbors_current = neighbors_start
            degree_current = degree_start
            while len(observed - sampled)> 0:               
                node_candidate = np.random.choice(list(neighbors_current))
                neighbors_candidate = set(self.__ego_network(node_current, 'ego_src'))
                observed = observed.union(neighbors_candidate)
                degree_candidate = float(len(neighbors_candidate))
                if degree_candidate == 0:
                    a = 1
                else:
                    a = min(1, degree_current/degree_candidate)
                if np.random.uniform() <= a:
                    sampled.add(node_candidate)
                    node_current = node_candidate
                    neighbors_current = neighbors_candidate
                    degree_current = degree_candidate
                    if self.args.undirected == 0:
                        if degree_candidate == 0:
                            node_current = node_start
                            neighbors_current = neighbors_start
                if len(sampled)>= self.args.sample_size:
                    break
        return list(sampled)
    
    
    def __greedy(self, target, map_node_group = None):            
        def candidate(u):
            def expand():
                D_t = np.zeros(D.shape)
                D_t[:, :idx+1] = D_u
                return D_t
            
            def harmonic():
                h_u = np.array([sum([1/d for d in D_u[r] if (d!=0) and not(d == np.float64('inf'))]) for r in range(D_u.shape[0])])
                h_o = np.array(h_original )
                ratio = h_u/h_o
                if map_node_group is None:
                    return(np.sum(ratio))
                else:
                    group_sigma = list()
                    for grp in map_group_node.keys():
                        group_sigma.append(np.sum(ratio[map_group_node[grp]])/len(map_group_node[grp]))
                    return(min(group_sigma))
        
            def update_distance(dist):
                #D_u[D_u == 0] = np.float64('inf')
                for i in range(D_u.shape[0]):
                    for j in range(D_u.shape[1]):
                        if i != j:
                            D_u[i ,j] = min(D_u[i,j], dist[i]+dist[j])
            D_u = D[:, :idx+1].copy()
            graph_u = graph.copy()
            edges_u = [(u, w) for w in self.__ego_network(u, 'ego_src') if w in graph.nodes()]
            graph_u.add_edges_from(edges_u)
            dist = np.zeros(D_u.shape[1])
            for node in graph_u.nodes():
                if node != u:
                    try:
                        dist[map_node_id[node]] = nx.shortest_path_length(graph_u, node,u)
                    except:
                        dist[map_node_id[node]] = np.float64('inf')
            D_u[:,idx] = dist[:D_u.shape[0]] 
            update_distance(dist)
            sigma = harmonic()
            D_u = expand()
            return sigma, D_u, graph_u        
        # initilization 
        D =np.zeros((len(target), self.args.sample_size))
        idx = len(target)-1
        map_node_id =  dict()
        graph = nx.Graph() # create the sampled graph 
        graph.add_nodes_from(target)
        neighbors, edges = set(), set()
        h_original = list()
        if not(map_node_group is None):
            map_group_node = dict()
        # create the induced graph of the target set, and set of the nodes in their neighborhood  
        for ct ,v in enumerate(target):
            neighbors_v = self.__ego_network(v, 'ego_src') 
            neighbors = neighbors.union(set(neighbors_v))
            edges_v = [(v, w) for w in neighbors_v if w in set(target)]
            edges = edges.union(set(edges_v))
            map_node_id[v] = ct
            if not(map_node_group is None):
                grp = map_node_group[v]
                if not(grp in map_group_node.keys()):
                    map_group_node[grp] = list()
                map_group_node[grp].append(map_node_id[v])
            h_original.append(self.__cent.loc[v,'harmonic'])
        graph.add_edges_from(list(edges)) 
        neighbors = list(neighbors -set(target))
        
        # calculate the shortest distance of the the induced subgraph. 
        for u in target:
            for v in target:
                if u != v:
                    try:
                        D[map_node_id[v], map_node_id[u]] = nx.shortest_path_length(graph, v,u)
                    except:
                        D[map_node_id[v], map_node_id[u]] =  np.float64('inf')

        # The main loop of algorithm
        while(idx <self.args.sample_size):
            idx += 1
            if idx == self.args.sample_size:
                break
            u_best =  neighbors[0]
            sigma_best, D_best, graph_best = candidate(u_best)
            for u in neighbors[1:]:
                sigma_u, D_u, graph_u = candidate(u)
                if sigma_u > sigma_best:
                    u_best = u
                    sigma_best, D_best, graph_best = sigma_u, D_u, graph_u           
            sigma, D, graph = sigma_best, D_best, graph_best 
            map_node_id[u_best] = idx
            neighbors = list(set(neighbors).union(set(self.__ego_network(u_best, 'ego_src'))) - set(graph.nodes()))
            
        return list(graph.nodes()) 
                  
    def __DFS(self, target):

        # randomly choose a node to start the walk, which is not a leaf(deadend).
        #users = pd.read_csv(self.args.feat_path, index_col=0 )
        #self.__load_data()
        sampled = dict()
        for node in target:
            sampled[node] = list()
            stack = list()
            stack.append(node)
            while len(stack)> 0:
                u = stack.pop()
                sampled[node].append(u)
                if len(set(sampled[node]))>= self.args.sample_size:
                    break
                for neighbor in np.unique(self.__ego_network(u, 'ego_src')):                  
                    if not(neighbor in sampled[node]):
                        stack.append(neighbor)
        sample = set(target)
        for idx in range(1, self.args.sample_size):
            for node in sampled.keys():
                try:
                    sample.add(sampled[node][idx])
                except:
                    pass
                if len(sample) == self.args.sample_size:
                        break   
            if len(sample) == self.args.sample_size:
                break   
        return list(sample) 
                
        
    def __BFS(self, target):
        # randomly choose a node to start the walk, which is not a leaf(deadend).
        #users = pd.read_csv(self.args.feat_path, index_col=0 )
        #self.__load_data()
        sampled = dict()
        for node in target:
            print(node)
            sampled[node] = list() 
            queue = list()
            queue.append(node)
            ct = 0
            while len(queue)> 0 and ct<self.args.sample_size*10 :
                u = queue.pop(0)
                sampled[node].append(u)
                #print(len(set(sampled[node])), end = ' ')
                if len(set(sampled[node]))>= self.args.sample_size:
                    break
                for neighbor in np.unique(self.__ego_network(u, 'ego_src')):  
                    if not(neighbor in sampled):
                        queue.append(neighbor)
                ct +=1
        sample = set(target)
        for idx in range(1, self.args.sample_size):
            for node in sampled.keys():
                try:
                    sample.add(sampled[node][idx])
                except:
                    continue
                if len(sample) == self.args.sample_size:
                    break   
            if len(sample) == self.args.sample_size:
                break   
        return list(sample) 
        
        
    def __original_graph_statistics(self):
        if path.exists(self.args.path + 'graph_info.txt'):
            if self.print_:
                print('>>> Loading the original graph statistics')
            self.org_info = pd.read_csv(self.args.path + 'graph_info.txt',index_col=0 ).to_dict()['0']
            if self.print_:
                tab_printer(self.org_info)
        else:
            if self.print_:
                print('>>> Calculating the original graph statistics')
            self.org_info = dict()
            #self.__load_data()
            n = self.__users.shape[0]
            map_node_to_id = dict()
            for ct, node in  enumerate(self.__users.index):
                map_node_to_id[node] = ct
            self.org_info['#nodes'] = n
            self.org_info['#edges'] = 0
            adj = sp.sparse.csr_matrix((n,n))
            for chunk in pd.read_csv(self.args.path + 'relations.csv', sep= self.args.sep, chunksize=self.args.chunksize):
                chunk = chunk.dropna(axis=0, how='any')
                self.org_info['#edges']+= chunk.shape[0]
                chunk['src'] = chunk['src'].apply(lambda x : map_node_to_id[x])
                chunk['dst'] = chunk['dst'].apply(lambda x : map_node_to_id[x])
                row = [v[0]for v in chunk.groupby(['src','dst']).size().index]
                col = [v[1]for v in chunk.groupby(['src','dst']).size().index]
                adj += sp.sparse.csr_matrix((chunk.groupby(['src','dst']).size().values, (row,col)), shape=(n, n))
                if self.args.undirected ==1:
                    adj += sp.sparse.csr_matrix((chunk.groupby(['src','dst']).size().values, (col,row)), shape=(n, n))
            r,_ = eigs(adj, k= 3, which= 'LR')    
            self.org_info['eig_l1'] = r[0].real
            self.org_info['eig_l2'] = r[1].real
            self.org_info['eig_l3'] = r[2].real
            if self.print_:
                tab_printer(self.org_info)
            pd.DataFrame.from_dict(self.org_info, orient='index').to_csv(self.args.path + 'graph_info.txt')
