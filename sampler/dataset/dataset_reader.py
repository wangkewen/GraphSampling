import io
import os
import numpy as np
import pandas as pd
import networkx as nx
import networkit as nk
from six.moves import urllib
from pathlib import Path

class GraphReader(object):
    r"""Class to read benchmark datasets for the sampling task.

    Args:
        dataset (str): Dataset of interest. One of musae_facebook_edges/github/deezer/lastfm. Default is 'musae_facebook_edges'.
    """

    def __init__(self, dataset: str = "musae_facebook_edges"):
        self.dataset = dataset
        self.datatype = ".txt"
        self.path = (
            "{}/graphdata/".format(Path.home())
        )
        self.sep = "\s+"
        self.node1 = "node1"
        self.node2 = "node2"

        if (self.dataset.endswith("com-dblp.ungraph") or
                self.dataset.endswith("com-amazon.ungraph") or
                self.dataset.endswith("com-youtube.ungraph") or
                self.dataset.endswith("loc-gowalla_edges") or
                self.dataset.endswith("loc-brightkite_edges") or
                self.dataset.endswith("email-Enron") or
                self.dataset.endswith("com-lj.ungraph")):
            self.sep = "\t"
            self.node1 = "FromNodeId"
            self.node2 = "ToNodeId"
        elif (self.dataset.endswith("musae_facebook_edges") or
                self.dataset.endswith("lastfm_asia_edges") or
                self.dataset.endswith("deezer_europe_edges") or
                self.dataset.endswith("musae_git_edges")):
            self.sep = ","
            self.datatype = ".csv"
            self.node1 = "node1"
            self.node2 = "node2"
        elif self.dataset.endswith("_sample"):
            self.path = "{}/sampletrace/".format(Path.home())
            self.sep = "\s+"
            self.datatype = ".txt"
            self.node1 = "node1"
            self.node2 = "node2"

    def _pandas_reader(self, path):
        """
        Reading file as a Pandas dataframe.
        """
        tab = pd.read_csv(
            path, encoding="utf8", sep=self.sep, comment="#", names=[self.node1, self.node2]
        )
        return tab

    def _dataset_reader(self):
        """
        Reading the dataset from the local graph file.
        """
        path = os.path.join(self.path, self.dataset + self.datatype)
        data = self._pandas_reader(path)
        return data

    def get_graph(self) -> nx.classes.graph.Graph:
        r"""Getting the graph.

        Return types:
            * **graph** *(NetworkX graph)* - Graph of interest.
        """
        data = self._dataset_reader()
        graph = nx.convert_matrix.from_pandas_edgelist(data, self.node1, self.node2)
        return graph
