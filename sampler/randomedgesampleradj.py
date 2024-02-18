import random
import networkx as nx
import networkit as nk
from typing import Union, List
from littleballoffur.edge_sampling import RandomEdgeSampler
import numpy as np

NKGraph = type(nk.graph.Graph())
NXGraph = nx.classes.graph.Graph


class RandomEdgeSamplerAdj(RandomEdgeSampler):
    r"""An implementation of random edge sampling AdjES.

    Args:
        number_of_edges (int): Number of edges. Default is 100.
        seed (int): Random seed. Default is 42.
    """

    def __init__(self, number_of_edges: int = 100, sample_by_edge : bool = True, seed: int = 42):
        self.number_of_edges = number_of_edges * 2
        self.sample_by_edge = sample_by_edge
        self.seed = seed
        self._set_seed()

    def _induce_graph_node(self, graph) -> Union[NXGraph, NKGraph]:
        """
        Inducing all of the edges given the sampled edges
        """
        nodes = set([node for edge in self._sampled_edges for node in edge])
        new_graph = self.backend.get_subgraph(graph, nodes)
        return new_graph

    def _induce_graph_edge(self, graph) -> Union[NXGraph, NKGraph]:
        """
        Inducing all of the edges given the sampled edges
        """
        p_number = 48
        all_edges_list = list(self.backend.get_edges(graph))
        nodedict = self._edges_map(all_edges_list)
        edges = set()
        nodes = set()
        while len(edges) < self.number_of_edges:
            edge_index_list = random.choices(all_edges_list, k=p_number)
            for edge in edge_index_list:
                for edge_node in edge:
                    if edge_node not in nodes:
                        neighbor_nodes_set = nodedict[edge_node] & nodes
                        for neighbor in neighbor_nodes_set:
                            edges.add((edge_node, neighbor))
                            edges.add((neighbor, edge_node))
                        nodes.add(edge_node)
                edges.add(edge)
                edges.add((edge[1], edge[0]))
        new_graph = self.backend.graph_from_edgelist(edges)
        return new_graph

    def sample(self, graph: Union[NXGraph, NKGraph]) -> Union[NXGraph, NKGraph]:
        """
        Sampling edges randomly with induction.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be sampled from.

        Return types:
            * **new_graph** *(NetworkX graph)* - The graph of sampled edges.
        """
        self._deploy_backend(graph)
        self._check_number_of_edges(graph)
        if self.sample_by_edge:
            new_graph = self._induce_graph_edge(graph)
        else:
            self._create_initial_edge_set(graph)
            new_graph = self._induce_graph_node(graph)
        return new_graph
