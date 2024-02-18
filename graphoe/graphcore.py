import networkit as nk
import networkx as nx
from littleballoffur.dataset import GraphReader
from littleballoffur.edge_sampling import RandomEdgeSampler
import numpy as np
import time
import os
from pathlib import Path

class GraphCore(object):
    r"""
    Sample graph, calculate graph attributes, and more

    Args: 
        graph: original graph
        datasource (str): file name of data source
    """

    def __init__(self, graph, datasource, sample_algo=None, sample_rate=None):
        self.graph = graph
        self.dataset = datasource
        self.sample_algo = sample_algo
        self.sample_rate = sample_rate
        self.sample_file_path = "{}/sampletrace/".format(Path.home())
        self.sample_file_type = ".txt"
        self.sample_sep = ' '
        self.sample_file_name = None
        self.evaluate_data_path = "{}/pevaldata/".format(Path.home())
        self.dist_data_path = "{}/pdistdata/".format(Path.home())

    def set_sample_file_path(self, sample_dir):
        """
        set sample file path using root path sample_dir
        """
        self.sample_file_path = "{}/{}/".format(Path.home(), sample_dir)

    def setSample_algo(self, sample):
        """
        set sample algo
        """
        self.sample_algo = sample

    def setSample_rate(self, sample_rate):
        """
        set sample rate
        """
        self.sample_rate = sample_rate

    def setDataset(self, datasource):
        """
        set dataset
        """
        self.dataset = datasource

    def initGraph(self):
        """
        generate and set graph
        """
        self.graph = self.generateGraph()
        return self.graph

    def generateGraph(self):
        """
        generate graph from reading a data file
        """
        graph = None
        if self.dataset:
            reader = GraphReader(self.dataset)
            graph = self._convertGraph(reader.get_graph())
        return graph

    def use_dist(self):
        """
        shift to use shift
        """
        self.evaluate_data_path = self.dist_data_path

    def write_eval(self, algolist, x_axis, ydict, measure):
        """
        write measure results of dataset into file
        """
        filename = self._evaldata_name(measure)
        with open("{}{}".format(self.evaluate_data_path, filename), "w") as fp:
            fp.write('#')
            fp.write(' '.join(algolist))
            fp.write('\n')
            fp.write(' '.join([str(e) for e in x_axis]))
            fp.write('\n')
            for algo in algolist:
                fp.write(' '.join([str(e) for e in ydict[algo]]))
                fp.write('\n')

    def read_eval(self, measure):
        """
        read measure from file

        @rtype: dict
        @return: dict of x,algos and ys
        """
        filename = self._evaldata_name(measure)
        results = dict()
        algos = None
        x = None
        with open("{}{}".format(self.evaluate_data_path, filename), "r") as fp:
            line = fp.readline()
            if line and line.startswith('#'):
                algos = line[1:].split()
            results['algos'] = algos
            line = fp.readline()
            results['x'] = [float(i) for i in line.split()]
            results['ys'] = dict()
            for y in algos:
                line = fp.readline()
                if not line:
                    break
                results['ys'][y] = [float(i) for i in line.split()]

        return results

    def _evaldata_name(self, measure):
        """
        file name of evaluation data
        """
        return "{}_{}.txt".format(self.dataset, measure)

    def _convertGraph(self, graph, nxtonk=True):
        """
        convert graph from nx to nk, and vice versa
        """
        if nxtonk:
            graph = nk.nxadapter.nx2nk(graph)
            self._remove_selfloops(graph)
            return graph
        else:
            return nk.nxadapter.nk2nx(graph)

    def _remove_selfloops(self, graph):
        """
        remove self loops
        """
        loop_count = graph.numberOfSelfLoops()
        if loop_count > 0:
            print("### self loop number:{}".format(loop_count))
            graph.removeSelfLoops()

    def writesamplegraph(self, graph=None):
        """
        write sample graph edges to text file
        """
        if not graph:
            graph = self.graph
        filepath = "{}{}{}".format(self.sample_file_path, self.sample_file_name, self.sample_file_type)
        with open(filepath, 'w+') as fp:
            for edge in graph.iterEdges():
                fp.write("{}{}{}\n".format(edge[0], self.sample_sep, edge[1]))

    def isexist(self, path, filename, filetype):
        """
        check if file path/filename exists
        """
        return os.path.isfile("{}{}{}".format(path, filename, filetype))

    def sampleGraph(self, sampler):
        """
        sample a graph from reading sample file or sample it
        """
        self.sample_file_name = "{}_{}_{}_sample".format(self.dataset, \
                self.sample_algo, self.sample_rate)

        samplegraph = None
        start = time.time()
        print("======sample start time {}======".format(start))
        if self.isexist(self.sample_file_path, self.sample_file_name, self.sample_file_type):
            self.dataset = self.sample_file_name
            samplegraph = self.generateGraph()
        else:
            samplegraph = sampler.sample(self.graph)
            self.writesamplegraph(samplegraph)
        end = time.time()
        
        print("======sample end time {}======".format(end))
        print(sampler)
        print("Sample Time(s): {:.2f}".format(end-start))
        print(samplegraph)
        return samplegraph

    def _tuple2dict(self, tupledata):
        """
        convert tuple data to dict()
        """
        ddict = dict()
        for node in tupledata:
            ddict[node[0]] = node[1]
        return ddict

    def _tuplefreq(self, tupledata):
        """
        convert tuple to dict(tuple[1], freq(tuple[1]))
        """
        freqdist = dict()
        for pair in tupledata[::-1]:
            if pair[1] not in freqdist:
                freqdist[pair[1]] = 0
            freqdist[pair[1]] += 1
        return freqdist

    def _constructDegreeCoeff(self, graph=None):
        """
        construct a sorted dict{degree:coeff}
        """
        if not graph:
            graph = self.graph
        clustercoeff_nk = nk.centrality.LocalClusteringCoefficient(graph)
        clustercoeff_nk.run()
        clusteringgraph = clustercoeff_nk.ranking()
        degree_nk = nk.centrality.DegreeCentrality(graph)
        degree_nk.run()
        degreegraph = self._tuple2dict(degree_nk.ranking())
        degreecoeff = dict()
        for pair in clusteringgraph:
            node = pair[0]
            degreecoeff[degreegraph[node]] = pair[1]
        return {e:degreecoeff[e] for e in sorted(degreecoeff.keys())}

    def _degreeDistribution(self, graph=None):
        """
        construct a sorted dict{degree:count}       
        """
        if not graph:
            graph = self.graph
        degreesdist = dict()
        degs = nk.centrality.DegreeCentrality(graph)
        degs.run()
        degdis = degs.ranking()
        degreesdist = self._tuplefreq(degdis)
        # degrees, nodes_count = np.unique(degdis, return_counts=True)
        return degreesdist

    def _coeffDistribution(self, graph=None):
        """
        construct a sorted dict{coeff:count}
        """
        if not graph:
            graph = self.graph
        clustercoeff_nk = nk.centrality.LocalClusteringCoefficient(graph)
        clustercoeff_nk.run()
        clusteringgraph = clustercoeff_nk.ranking()
        coeffdist = self._tuplefreq(clusteringgraph)
        return coeffdist

    def _kcore(self, graph=None):
        """
        construct a core dist{corenumber:count}
        """
        if not graph:
            graph = self.graph
        corede = nk.centrality.CoreDecomposition(graph)
        corede.run()
        cores = corede.scores()
        cores.sort()
        coredist = dict()
        for k in cores:
            k = int(k)
            if k not in coredist:
                coredist[k] = 0
            coredist[k] += 1
        return coredist

    def _pagerank(self, graph=None):
        """
        calculate pagerank for each node {node:pr}
        """
        if not graph:
            graph = self.graph
        pr = nk.centrality.PageRank(graph)
        pr.norm = nk.centrality.Norm.L1_NORM
        pr.maxIterations = 100
        pr.run()
        nodepr = self._tuple2dict(pr.ranking())
        return nodepr

    def _betweenness(self, graph=None):
        """
        construct a betweenness dist{betweenness:count}
        """
        if not graph:
            graph = self.graph
        bc = nk.centrality.Betweenness(graph, normalized=True)
        bc.run()
        bcdist = bc.ranking()
        betweendist = self._tuplefreq(bcdist)
        return betweendist

    def _closeness(self, graph=None):
        """
        construct a closeness dist{closeness:count}
        """
        if not graph:
            graph = self.graph
        close = nk.centrality.Closeness(graph, False, nk.centrality.ClosenessVariant.Generalized)
        close.run()
        closedist = close.ranking()
        closenessdist = self._tuplefreq(closedist)
        return closenessdist

    def _eigenvectorcent(self, graph=None):
        """
        construct a eigenvector centrality dist{eigenvector_score:count}
        """
        if not graph:
            graph = self.graph
        eigen = nk.centrality.EigenvectorCentrality(graph)
        eigen.run()
        eigendist = eigen.ranking()
        eigenvcdist = self._tuplefreq(eigendist)
        return eigenvcdist
    
    def _globalclusteringcoeff(self, graph=None):
        """
        calculate global clustering coefficient
        """
        if not graph:
            graph = self.graph
        gcc = nk.globals.clustering(graph)
        return gcc

    def _density(self, graph=None):
        """
        calculate density
        """
        if not graph:
            graph = self.graph
        ds = nk.graphtools.density(graph)
        return ds

    def _maxkcore(self, graph=None):
        """
        calculate max core number of k-core
        """
        if not graph:
            graph = self.graph
        mc = nk.centrality.CoreDecomposition(graph)
        mc.run()
        mc = mc.maxCoreNumber()
        return mc

    def _diameter(self, graph=None):
        """
        calculate graph diameter
        """
        if not graph:
            graph = self.graph
        dia = nk.distance.Diameter(graph,algo=1)
        dia.run()
        dia = dia.getDiameter()[0]
        return dia

    def _avgdegree(self, graph=None):
        """
        calculate average degree
        """
        if not graph:
            graph = self.graph
        avg = (2.0 * graph.numberOfEdges()) / graph.numberOfNodes()
        return avg

    def _transitivity(self, graph=None):
        """
        calculate transitivity
        """
        if not graph:
            graph = self.graph
        graph = self._convertGraph(graph, False)
        trans = nx.transitivity(graph)
        return trans

    def construct(self, measure, graph=None):
        """
        construct dict of degreeCoeff or degreeDist
        """
        if measure == 'degreecoeff':
            return self._constructDegreeCoeff(graph)
        elif measure == 'degreedist':
            return self._degreeDistribution(graph)
        elif measure == 'coeffdist':
            return self._coeffDistribution(graph)
        elif measure == 'kcoredist':
            return self._kcore(graph)
        elif measure == 'betweennessdist':
            return self._betweenness(graph)
        elif measure == 'closenessdist':
            return self._closeness(graph)
        elif measure == 'eigenvectorcent':
            return self._eigenvectorcent(graph);
        elif measure == 'globalclusteringcoeff':
            return self._globalclusteringcoeff(graph)
        elif measure == 'density':
            return self._density(graph)
        elif measure == 'maxkcore':
            return self._maxkcore(graph)
        elif measure == 'diameter':
            return self._diameter(graph)
        elif measure == 'avgdegree':
            return self._avgdegree(graph)
        elif measure == 'transitivity':
            return self._transitivity(graph)
