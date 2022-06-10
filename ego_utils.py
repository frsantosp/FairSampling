import pandas as pd
import numpy as np
import os


def random_walk(path, walk_len =1000):
    '''
    generate random walks on graph.
    input:
    path: input data directory
    walk_len(int): length of random walk
    '''
    # randomly choose a node to start the walk, which is not a leaf(deadend).
    node_start = int(np.random.choice(os.listdir(path+ 'ego_src/')).split('.')[0])
    while deadend(node_start, path):
        node_start = int(np.random.choice(os.listdir(path+ 'ego_src/')).split('.')[0])
    node_c = node_start # current node
    walk = list()    
    for i in range(walk_len): # main loop of the walk
        walk.append(node_c)
        node_c = np.random.choice(get_ego_src(node_c,path)) # randomly choose one of the neighbors of current node.
        if deadend(node_c, path): # if the new node is deadend start over from start node.
            walk.append(node_c)
            node_c = node_start
    return walk

def deadend(node, path):
    # check if the input node is src of any edge that is not self loop.
    # return True if it's a dead end, false ow
    output = True
    neighbors = get_ego_src(node,path)
    if len(neighbors)> 0:
        if len(np.unique(neighbors)) > 1:
            output = False
        elif np.unique(neighbors)[0]!= node:
            output = False       
    return output   

def get_ego_src(node, path ):
    # return the list of all the edges(src,dst) such that src= node. 
    output = list()
    try:
        data = pd.read_csv(path+'ego_src/'+str(node)+'.ego', names=['dst', 'day', 'type', 'time_ms'])
        output.extend(list(data.dst))
    except:
        pass
#     try:
#         data = pd.read_csv(path+'ego_src/'+str(node)+'.0.ego', names=['dst', 'day', 'type', 'time_ms'])
#         output.extend(list(data.dst))
#     except:
#         pass
    return [int(node) for node in output]

def get_ego_dst(node, path ):
    # return the list of all the edges(src,dst) such that dst= node. 
    output = list()
    try:
        data = pd.read_csv(path+'ego_dst/'+str(node)+'.ego', names=['src', 'day', 'type', 'time_ms'])
        output.extend(list(data.src))
    except:
        pass
#     try:
#         data = pd.read_csv(path+'ego_dst/'+str(node)+'.0.ego', names=['src', 'day', 'type', 'time_ms'])
#         output.extend(list(data.src))
#     except:
#         pass
    return [int(node) for node in output] 

def all_nodes(path):
    files = os.listdir(path+ 'ego_src/')
    nodes_src = [int(file.split('.')[0]) for file in files if len(file.split('.')[0])>0 ]
    files = os.listdir(path+ 'ego_dst/')
    nodes_dst = [int(file.split('.')[0]) for file in files if len(file.split('.')[0])>0]
    nodes = np.unique(nodes_src + nodes_dst)
    return sorted(nodes)

def ego_edges_df(w, nodes, path_e):
    edges_w = pd.DataFrame( columns =['src', 'dst'] )
    srcs = list()
    dsts = list()
    for node in get_ego_src(w,path_e):
        if node in nodes:
            srcs.append(w)
            dsts.append(node)
    for node in get_ego_dst(w,path_e):
        if node in nodes:
            dsts.append(w)
            srcs.append(node)
    edges_w['src'] = srcs 
    edges_w['dst'] = dsts
    return edges_w
