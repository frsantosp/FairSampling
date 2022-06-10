import pandas as pd
import os
from os import path
import numpy as np
import networkx as nx
from utils import *

class Evaluator_nx:
    def __init__(self, args, print_ = True):
        self.args = args
        self.print_ = print_
        self.__load_data()
        self.__original_graph_statistics()
        self.__load_evaluations()
        self.__samples_dirctor = self.args.sample_path + self.args.path.split('./data/')[1]
    
    def sample_list(self):        
        files = os.listdir(self.__samples_dirctor)
        files =[file for file in files if '.txt' in file]
        return files
    
    def __evaluation_file(self):
        path = './evaluations/'
        path += self.args.path.split('./data/')[1].strip('/').replace('/','_')+ '.csv' 
        return path
    
    def __load_evaluations(self):
        if path.exists(self.__evaluation_file()):
            self.__results = pd.read_csv(self.__evaluation_file(), index_col = 0)
        else:
            self.__results = pd.DataFrame(columns=['sample_id','sampler', 'sample_size'])

    def load_sample(self, sample):
        nodes = open(self.__samples_dirctor +sample , 'r').readline().strip()
        nodes = [int(node) for node in nodes.split(' ')]
        return nodes  
    
    def save_evaluations(self):
        self.__results.to_csv(self.__evaluation_file())
    
    def __load_data(self):
            #self.__edges = pd.read_csv(self.args.path + 'relations.csv')[['src', 'dst']]
            #self.__users = pd.read_csv(self.args.path + 'usersdata.csv' ).set_index('user_id')
            #self.cents = pd.read_csv(self.args.path + 'node_centralities.csv', index_col=0)
            self.__edges = pd.read_pickle(self.args.path + "relations.pkl")
            self.__cent = pd.read_pickle(self.args.path + "node_centralities.pkl")
            self.__users = pd.read_pickle(self.args.path + "usersdata.pkl")
            
    def __original_graph_statistics(self):
        if self.print_:
            print('>>> Loading the original graph statistics')
        self.org_info = pd.read_csv(self.args.path + 'graph_info.txt',index_col=0 ).to_dict()['0']
        if self.print_:
            tab_printer(self.org_info)
            
            
    def __induced_subgraph(self, nodes):
        return self.__edges[self.__edges.src.isin(nodes) & self.__edges.dst.isin(nodes)] 

    def evaluate_sample(self, sample, to_results = True, nodes = None):
        if nodes is None:
            nodes = self.load_sample(sample)
        edges = self.__induced_subgraph(nodes)
        edges.columns = ['source', 'target']
        graph = nx.from_pandas_edgelist(edges)
        for node in set(nodes)-set(graph.nodes):
            graph.add_node(node)
        cents = pd.DataFrame()
        try:
            cents['eigenvector'] = pd.Series(nx.eigenvector_centrality(graph))
        except:
            pass
        cents['closeness'] = pd.Series(nx.closeness_centrality(graph))
        try:
            cents['information'] = pd.Series(nx.information_centrality(graph))
        except:
            pass
        cents['pagerank'] = pd.Series(nx.pagerank(graph, alpha=0.85))
        cents['betweenness'] = pd.Series(nx.betweenness_centrality(graph))
        cents['harmonic'] =  pd.Series(nx.harmonic_centrality(graph))
        cents['degree'] = pd.Series(dict(graph.degree))
        if to_results:
            row = {'sample_id':sample ,'sampler':sample.split('_')[0], 'sample_size': int(sample.split('_')[1])}
            file_g = open('.'+self.__evaluation_file().strip('.csv')+'_'+str(row['sample_size'])+'_gender.csv','a')
            file = open('.'+self.__evaluation_file().strip('.csv')+'_'+str(row['sample_size'])+'.csv','a')
            for col in cents.columns:
                line_g = row['sample_id']+','+ row['sampler']+ ',' + col
                line = row['sample_id']+','+ row['sampler']+ ',' + col
                df = cents[col].loc[nodes]/self.cents[col].loc[nodes]
                row[col] = np.sum(df )/len(nodes)
                for x in self.__users.loc[df.index[np.argsort(df.values)], 'gender'].values:
                    line_g += ',' + str(x)
                for x in sorted(df.values):   
                    line += ',' + str(x)
                file_g.write(line_g+'\n')
                file.write(line+'\n')
            file_g.close()
            file.close()
            self.__results = self.__results.append(row, ignore_index=True)
        else:
            return cents, graph
    
