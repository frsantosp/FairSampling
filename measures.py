import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
from scipy.sparse.linalg import eigsh, eigs
from utils import tab_printer

class measures():
    def __init__(self, args, print_):
        self.args = args
        self.result = dict()
        self.print_ = print_
        #self.__original_graph_statistics()
    
    def evalute_measure(self, edges, nodes, which):
        self.__sample_features( edges, nodes)
        if which == 'eigs':
            r = self.__eigs(3,'LR')
            l = (self.org_info['eig_l1'] - r[0].real)**2
            l += (self.org_info['eig_l2'] - r[1].real)**2
            l += (self.org_info['eig_l3'] - r[2].real)**2
            return np.sqrt(l)
        elif 'degree' in which:
            self.degree_evaluation(which)
            return self.result[which]
        elif 'nCGRtopk' in which:
            self.__nCGRtopk_evaluation('out_degree')
            return self.result['out_degree_nCGRtopk']
    
    def evaluate_all(self, edges, nodes):           
        self.__sample_features( edges, nodes)
        self.__eigs(5)
        self.degree_evaluation()
        self.__representativity_evaluation()
        #self.__information_loss() 
        self.__edge_parity_evaluation(edges)
        self.__nCGRtopk_evaluation()
        if self.print_:
            self.__print_result()
    
    def __sample_features(self, edges, nodes):
        self.adj = self.genereate_adj( edges.copy(), nodes)
        self.feats = self.__sample_degrees( nodes)
        
        
    def genereate_adj(self, edges, nodes):
        self.node_to_id = dict()
        n = len(nodes)
        for ct, node in enumerate(nodes):
            self.node_to_id[node] = ct
        edges['src'] = edges['src'].apply(lambda x :self.node_to_id[x])
        edges['dst'] = edges['dst'].apply(lambda x :self.node_to_id[x])
        row = [v[0]for v in edges.groupby(['src','dst']).size().index]
        col = [v[1]for v in edges.groupby(['src','dst']).size().index]
        adj = sp.sparse.csr_matrix((edges.groupby(['src','dst']).size().values, (row,col)), shape=(n, n),dtype=float)
        return adj
    
    def __KL_divergence(self, p, q):
        return stats.entropy(p,q)
    
    def __D_statistic(self, p,q):
        return np.max(np.abs(np.cumsum(p)-np.cumsum(q)))

    
    def __sample_degrees(self, nodes):
        '''
        given an induced subgraph calculated the degree sequence 
        input: 
            1. edges: dataframe of edge lists.
            2. self.args.feat_path: the features file address.
        output:
            1. dataframe of featuers.
        '''
        n = len(nodes)
        nodes_f = pd.read_pickle(self.args.path + 'usersdata.pkl' )
        nodes_f['sample_out_degree'] = np.zeros(nodes_f.shape[0])
        nodes_f['sample_in_degree'] = np.zeros(nodes_f.shape[0])
        nodes_f.loc[nodes,'sample_out_degree'] = np.array(self.adj.sum(axis=1))
        nodes_f.loc[nodes,'sample_in_degree'] = np.array(self.adj.sum(axis=0).reshape((n,1)))
        return nodes_f
    
    def __eigs(self, k, which = 'both'):
        if which == 'both':
            eigenvalues_l, eigenvectors = eigs(self.adj, k= k, which= 'LR')
            #print('LA',eigenvalues)
            eigenvalues_s, eigenvectors = eigs(self.adj, k= k, which= 'SR')
            #print('SA',eigenvalues)
            return eigenvalues_s, eigenvalues_l
        else:
            eigenvalues, eigenvectors = eigs(self.adj, k= k, which = which)
            return eigenvalues
    
    def __conditional_entropy(self, d,x):
        return  np.sum(-np.log(d[d>0]/x)*d[d>0])
    
    def __information_loss(self):
        def p(prt_val, d, degree_type ):
            idx = (self.feats[self.args.protected] == prt_val) & (self.feats[degree_type] == d)
            return np.sum(idx)/self.feats.shape[0]

        def p_sample(prt_val, d, degree_type):
            idx_s = (self.feats['sample_in_degree']>0) | (self.feats['sample_out_degree']>0)
            idx = (self.feats[self.args.protected] == prt_val) & (self.feats[degree_type] == d) & idx_s
            return np.sum(idx)/np.sum(idx_s)
        joint_dist = dict()
        sample_joint_dist = dict()
        for val in self.feats[self.args.protected].unique():
            for degree_type in ['in_degree', 'out_degree']:
                joint_dist[(val,degree_type)] = list()
                sample_joint_dist[(val,degree_type)] = list()
                for d in range(int(self.feats[degree_type].max())+1):
                    joint_dist[(val,degree_type)].append(p(val, d,degree_type ))
                    sample_joint_dist[(val,degree_type)].append(p_sample(val, d,'sample_'+ degree_type ))
                joint_dist[(val,degree_type)] = np.array(joint_dist[(val,degree_type)])
                sample_joint_dist[(val,degree_type)] = np.array(sample_joint_dist[(val,degree_type)])
        for val in self.feats[self.args.protected].unique():
            for degree_type in ['in_degree', 'out_degree']:
                x = np.sum(self.feats[self.args.protected] == val)/self.feats.shape[0]
                print(val, degree_type, self.__conditional_entropy(joint_dist[(val,degree_type)],x))
                idx_s = (self.feats['sample_in_degree']>0) | (self.feats['sample_out_degree']>0)
                x = (np.sum(self.feats[self.args.protected] == val) & idx_s)/np.sum(idx_s)
                print(val, 'sample_'+ degree_type, self.__conditional_entropy(sample_joint_dist[(val,degree_type)],x))
                
                
                
    def __nCGRtopk_evaluation(self,which ='all', k = 100):
        n = self.feats[(self.feats.sample_in_degree>0) |(self.feats.sample_out_degree>0) ].shape[0]
        if which == 'all':
            cols = ['in_degree', 'out_degree']
        else:
            cols = [which]
        for col in cols:  
            idx = np.argsort(-1*self.feats[col].values)
            self.feats.loc[self.feats.index[idx], 'rank'] =  np.array(range(1,self.feats.shape[0]+1))
            self.feats['rel']=  (1/self.feats['rank'].values)/np.sum(self.feats['rank'])
            idxs = np.argsort(-1*self.feats['sample_'+col].values)
            self.feats.loc[self.feats.index[idxs], 'rank_s'] =  np.array(range(1,self.feats.shape[0]+1))
            self.feats['rel_s']=  2*(1/self.feats['rank_s'].values)/(n*(n-1))
            vals = list()
            for val in self.feats[self.args.protected].unique():
                org = np.sum(self.feats.loc[self.feats.index[idx]][:k][self.feats.loc[self.feats.index[idx]][:k][self.args.protected] == val]['rel'])
                orgs =np.sum(self.feats.loc[self.feats.index[idxs]][:k][self.feats.loc[self.feats.index[idxs]][:k][self.args.protected] == val]['rel_s'])
                vals.append(orgs/org)
            self.result[col + '_nCGRtopk'] = self.__normalized_var( vals)
        
    
    def __representativity_evaluation(self):
        '''
        Proposed fariness measure
        '''
        reps_out = list()
        reps_in = list()
        for name in self.feats[self.args.protected].unique():
            idx = (self.feats.out_degree > 0) & (self.feats[self.args.protected] == name)
            #reps_out[name] = np.sum(self.feats[idx].sample_out_degree/self.feats[idx].out_degree)/self.feats[idx].shape[0]
            reps_out.append(np.sum(self.feats[idx].sample_out_degree/self.feats[idx].out_degree)/self.feats[idx].shape[0])
            idx = (self.feats.in_degree > 0) & (self.feats[self.args.protected] == name)
            #reps_in[name] =  np.sum(self.feats[idx].sample_in_degree/self.feats[idx].in_degree)/self.feats[idx].shape[0]
            reps_in.append( np.sum(self.feats[idx].sample_in_degree/self.feats[idx].in_degree)/self.feats[idx].shape[0])
        self.result['in_degree_rep_bias'] = self.__normalized_var(reps_in )
        self.result['out_degree_rep_bias'] = self.__normalized_var(reps_out )
   
    def __edge_parity_evaluation(self, edges):
        #def edge_parity(edges , path_f, feat = 'sex'):
        edge_types = list()
        def node_type(node):
            return self.feats.loc[node,self.args.protected]
        x = edges['src'].apply(node_type)
        y = edges['dst'].apply(node_type)
        for val1 in self.feats[self.args.protected].unique():
            for val2 in self.feats[self.args.protected].unique():
                edge_types.append( np.sum((x == val1) & (y==val2)))
        self.result['edge_parity_bias'] = self.__normalized_var(edge_types)
        
    def __normalized_var(self, x):
        x = np.array(x)
        x = x/np.linalg.norm(x)
        return np.var(x)   
    
    def __degree_dist(self, x):
        x = x.value_counts()/np.sum(x.value_counts())
        max_degree = int(np.max(x.index))
        p = np.zeros(max_degree+1)
        for id in sorted(x.index):
            p[int(id)] = x[id]
        return p
    
    def degree_evaluation(self, which = 'all'):
        p = self.__degree_dist(self.feats.out_degree )
        q = self.__degree_dist(self.feats.sample_out_degree)
        if len(p)> len(q):
            q =np.concatenate((q,np.zeros(len(p)-len(q))))
        if (which == 'all') or ('out_degree_KL'):
            self.result['out_degree_KL'] = self.__KL_divergence( q, p)
        if (which == 'all') or ('out_degree_DS'):
            self.result['out_degree_DS'] = self.__D_statistic( p, q)
        p = self.__degree_dist(self.feats.in_degree)
        q = self.__degree_dist(self.feats.sample_in_degree)
        if len(p)> len(q):
            q =np.concatenate((q,np.zeros(len(p)-len(q))))
        if (which == 'all') or ('in_degree_KL'):
            self.result['in_degree_KL'] = self.__KL_divergence( q, p)
        if (which == 'all') or ('in_degree_DS'):
            self.result['in_degree_DS'] = self.__D_statistic( p, q)

    #def __ClusterCoefficient(self):    
        
    def __print_result(self):
        tab_printer(self.result)
        
    def __original_graph_statistics(self):
        self.org_info = pd.read_csv(self.args.path + 'graph_info.txt',index_col=0 ).to_dict()['0']
