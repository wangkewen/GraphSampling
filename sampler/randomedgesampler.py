import random
import networkx as nx
import networkit as nk
from typing import Union, List
from littleballoffur.sampler import Sampler
import numpy as np

NKGraph = type(nk.graph.Graph())
NXGraph = nx.classes.graph.Graph


class RandomEdgeSampler(Sampler):
    r"""An implementation of random edge sampling.

    Args:
        number_of_edges (int): Number of edges. Default is 100.
        seed (int): Random seed. Default is 42.
    """

    def __init__(self, number_of_edges: int = 100, seed: int = 42):
        self.number_of_edges = number_of_edges
        self.seed = seed
        self._set_seed()

    def _edges_map(self, edges):
        """
        Generate dict[node1, set(node2,node3,...)] for edges
        """
        nodedict = dict()
        for edge in edges:
            if edge[0] not in nodedict:
                nodedict[edge[0]] = set()
            nodedict[edge[0]].add(edge[1])
            if edge[1] not in nodedict:
                nodedict[edge[1]] = set()
            nodedict[edge[1]].add(edge[0])
        return nodedict

    def _select_nodes_by_edge(self, graph: Union[NXGraph, NKGraph], node_list: List[int], weight_list: List[float]) -> List[int]:
        """
        select nodes/edges until reach induced edge number
        """
        nodedict = self._edges_map(self.backend.get_edges(graph))
        edges = set()
        nodes = set()
        p_number = 48
        while len(edges) < self.number_of_edges:
            sampled_list = random.choices(node_list, weights=weight_list, k=p_number)
            for sampled_node in sampled_list:
                if sampled_node not in nodes:
                    neighbor_nodes_set = nodedict[sampled_node] & nodes
                    for neighbor in neighbor_nodes_set:
                        edges.add((sampled_node, neighbor))
                        edges.add((neighbor, sampled_node))
                nodes.add(sampled_node)
        return edges

    def _create_initial_edge_set(self, graph):
        """
        Choosing initial edges.
        """
        edges = list(self.backend.get_edges(graph))
        self._sampled_edges = random.sample(edges, self.number_of_edges)

    def sample(self, graph: Union[NXGraph, NKGraph]) -> Union[NXGraph, NKGraph]:
        """
        Sampling edges randomly.

        Arg types:
            * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.

        Return types:
            * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled edges.
        """
        self._deploy_backend(graph)
        self._check_number_of_edges(graph)
        self._create_initial_edge_set(graph)
        new_graph = self.backend.graph_from_edgelist(self._sampled_edges)
        return new_graph
