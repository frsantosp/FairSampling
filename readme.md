#Topological and Fairness-Aware Graph Sampling for Network Analysis

Network sampling is the task of selecting a subset of nodes and links from a network in a way that preserves its topological
properties and other user requirements. This paper investigates the problem of generating an unbiased network sample that contains
balanced proportion of nodes from different groups. Creating such a representative sample would require handling the trade-off
between ensuring structural preservability and group representativity of the selected nodes. We present a novel max-min subgraph
fairness measure that can be used as a unifying framework that combines both criteria. A greedy algorithm is then proposed
to generate a fair and representative sample from an initial set of target nodes. A theoretical approximation guarantee
for the output of the proposed greedy algorithm based on submodularity and curvature ratios is also presented.
Experimental results on real-world datasets show that the proposed method will generate more fair and representative samples compared to other existing network sampling methods.

##Datasets

The experiments were ran on four different real world datasets. The datasets used were Facebook, Tagged, Credit and German. All the datasets are in pickle files in the github.

##Usage:

Examples to run the sampling:

Script 1: Get sample graph using Fair Random Walk(FRW) on a undirected graph on the German dataset.
```python
python run_sampling_targeted.py --ego-exist 0 --path '.German/'  --sampler FRW  --undirected 1 --sample-number 10 --random-target 1
```

Script 2: Get sample graph using Breadth First Search on a undirected graph on the credit dataset.
```Python
python run_sampling_targeted.py --ego-exist 0 --path './data/Credit/' --protected Age  --sampler BFS  --undirected 1 --sample-number 1 --random-target 0
```

Script 3: Get sample graph using Greedy Fair Sampling on a undirected graph on the German dataset.
```python
python run_sampling_targeted.py --ego-exist 0 --path './data/German/'  --sampler GF  --undirected 1 --sample-number 10 --random-target 0
```

