import random
import networkx as nx
import networkit as nk
from typing import Union
from littleballoffur.edge_sampling import RandomEdgeSampler
import numpy as np
import math
import time
from multiprocessing.pool import Pool
from multiprocessing import Queue, Process


NKGraph = type(nk.graph.Graph())
NXGraph = nx.classes.graph.Graph

class RandomEdgeSamplerAdp(RandomEdgeSampler):
    r"""An implementation of random edge sampling AdpES.

    Args:
        p (float): Sampling probability. Default is 0.1.
        seed (int): Random seed. Default is 42.
    """

    def __init__(self, number_of_edges: int = 100, p: float = 0.1, seed: int = 42):
        self.p = p
        self.number_of_edges = number_of_edges * 2
        self.seed = seed
        self._set_seed()

    def _create_initial_set(self, graph):
        """
        Creatin an initial edge and node set and a reshuffled edge stream.
        """

        ti1 = time.time()
        self._all_nodes_degree = {node : self.backend.get_degree(graph, node) for node in self.backend.get_nodes(graph)}
        self._all_edges_count = self.backend.get_number_of_edges(graph)
        self._all_nodes_count = self.backend.get_number_of_nodes(graph)

        self._avgdegree = (self._all_edges_count * 2.0) / self._all_nodes_count
        self._sigma = 0.02

        self._order_edge = list(self.backend.get_edges(graph))
        self._order_index = list()
        self._order_weight = []
        self._alpha_list = []

        alpha = 2.0
        delta = 0.25

        self._weight_size = int((alpha * 2) / delta) + 1
        ti2 = time.time()

        print("### initial startup time: {}".format(ti2-ti1))
        edge_deg_list = []
        for i in range(len(self._order_edge)):
            edge = self._order_edge[i]
            edge_deg_list.append((self._all_nodes_degree[edge[0]] + self._all_nodes_degree[edge[1]]) / 2)

        ta = time.time()

        for i in range(self._weight_size):
            self._alpha_list.append(alpha)
            orderweights = []
            k = 0
            all_weight = 0.0
            for edge_deg in edge_deg_list:
                one_weight = edge_deg ** alpha
                all_weight += one_weight
                orderweights.append(all_weight)
                k += 1
            self._order_weight.append(orderweights)
            alpha -= delta
        tb = time.time()
        print("### total weight calculation time: {}, loop time: {}".format(tb - ti2, tb - ta))

    def _sample_edges(self):
        """
        Inducing all of the edges given the sampled edges
        """
        p_number = 48
        all_edges_list = self._order_edge
        nodedict = self._edges_map(all_edges_list)
        edges = set()
        nodes = set()
        weightindex = self._weight_size // 2
        sampleavgde = 0.0
        avgalpha = 0.0
        count = 0
        accrandomtime = 0.0
        while len(edges) < self.number_of_edges:
            count += 1
            avgalpha += self._alpha_list[weightindex]
            t1 = time.time()
            edge_index_list = random.choices(self._order_edge, cum_weights=self._order_weight[weightindex], k=p_number)
            t2 = time.time() - t1
            accrandomtime += t2
            for edge in edge_index_list:
                if edge in edges:
                    continue
                for edge_node in edge:
                    if edge_node not in nodes:
                        neighbor_nodes_set = nodedict[edge_node] & nodes
                        for neighbor in neighbor_nodes_set:
                            edges.add((edge_node, neighbor))
                            edges.add((neighbor, edge_node))
                        nodes.add(edge_node)
                edges.add(edge)
                edges.add((edge[1], edge[0]))
            # adjust weight
            sampleavgde = (1.0 * len(edges)) / (len(nodes) if len(nodes) > 0 else 1)
            diffratio = (sampleavgde - self._avgdegree) / (self._avgdegree if self._avgdegree != 0 else 1)
            time_ns = time.time()
            print("edge#={} timestamp={} alpha={} avgdeg={} degdiff={} randomtime={}".format(self.number_of_edges, time_ns, self._alpha_list[weightindex], sampleavgde, diffratio, t2))
            if abs(diffratio) < self._sigma:
                continue
            elif diffratio > 0:
                if weightindex < self._weight_size - 1:
                    weightindex += 1
            else:
                if weightindex > 0:
                    weightindex -= 1

        avgalpha = avgalpha / (count if count > 0 else 1)
        print("########### Y avg degree:{}, avg alpha:{} totalrandomtime:{}".format(self._avgdegree, avgalpha, accrandomtime))
        tp1 = time.time()
        new_graph = self.backend.graph_from_edgelist(edges)
        tp2 = time.time()
        print("### graph time:{}".format(tp2-tp1))

        return new_graph

    def sample(self, graph: Union[NXGraph, NKGraph]) -> Union[NXGraph, NKGraph]:
        """
        Sampling edges randomly with induction.

        Arg types:
            * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.

        Return types:
            * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled edges.
        """
        self._deploy_backend(graph)
        self._check_number_of_edges(graph)
        ta = time.time()
        self._create_initial_set(graph)
        tb = time.time()
        print("### initial time:{}".format(tb-ta))
        new_graph = self._sample_edges()
        tc = time.time()
        print("### sample time:{}".format(tc-tb))
        return new_graph

