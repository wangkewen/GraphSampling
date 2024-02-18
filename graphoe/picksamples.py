import networkit as nk
from littleballoffur.node_sampling import RandomNodeSampler, DegreeBasedSampler
from littleballoffur.edge_sampling import RandomEdgeSampler, RandomEdgeSamplerAdj, RandomEdgeSamplerWithPartialInduction, RandomEdgeSamplerX, HybridNodeEdgeSampler, RandomEdgeSamplerAdp, RandomEdgeSamplerGSES, RandomEdgeSamplerSUBM
from littleballoffur.exploration_sampling import RandomWalkSampler, RandomWalkWithRestartSampler, RandomWalkWithJumpSampler, BreadthFirstSearchSampler, ForestFireSampler
import math
import time
import datetime
from metricsprocessor import MetricsProcessor
from graphcore import GraphCore
from multiprocessing import Queue
from multiprocessing import Process
import multiprocessing
from pathlib import Path
import matplotlib.pyplot as plt

def sampleset(gc, n, samplepath=None):
    """
    obtain sample graph from file or generate sample
    """

    if samplepath:
        gc.set_sample_file_path(samplepath)
    sampler = None
    randomname = gc.sample_algo
    if randomname == 'X':
        sampler = gc.sampleGraph(RandomEdgeSamplerX(n))
    elif randomname == 'Y':
        sampler = gc.sampleGraph(RandomEdgeSamplerAdp(n))
    elif randomname == 'ESI':
        sampler = gc.sampleGraph(RandomEdgeSamplerAdj(n))
    elif randomname == 'PESI':
        sampler = gc.sampleGraph(RandomEdgeSamplerWithPartialInduction(n))
    elif randomname == 'RW':
        sampler = gc.sampleGraph(RandomWalkSampler(n))
    elif randomname == 'NS':
        sampler = gc.sampleGraph(RandomNodeSampler(n))
    elif randomname == 'GSES':
        sampler = gc.sampleGraph(RandomEdgeSamplerGSES(n))
    elif randomname == 'SUBM':
        sampler = gc.sampleGraph(RandomEdgeSamplerSUBM(n))
    elif randomname == 'NDS':
        sampler = gc.sampleGraph(DegreeBasedSampler(n))
    else:
        sampler = None
    return sampler

def samplegraphs(gc, samples, samplepath=None):
    """
    generate sample graphs in sample rate = sample_rate
    """
    #samples = ['Y', 'X', 'ESI', 'PESI', 'RW', 'NS', 'GSES', 'SUBM', 'NDS']
    #samples = ['SUBM']
    sample_perc = gc.sample_rate
    n = math.floor(gc.graph.numberOfEdges() * sample_perc)
    print("n={}".format(n))

    # generate sample graphs
    sampleps = list()
    for randomname in samples:
        gc.setSample_algo(randomname)
        p = Process(target=sampleset, args=(gc, n, samplepath,))
        sampleps.append(p)
        p.start()

    for p in sampleps:
        p.join()

def generate_batch_samples(datasource, algos, sample_rates=[0.1, 0.2, 0.3, 0.4, 0.5], samplepath=None):
    """
    generate a batch of sample data
    """
    
    # datasource = 'musae_facebook_edges'
    # datasource = 'musae_git_edges'
    # datasource = 'deezer_europe_edges'
    # datasource = 'lastfm_asia_edges'

    # datasource = 'email-Enron'
    # datasource = 'loc-brightkite_edges'
    # datasource = 'loc-gowalla_edges'

    # datasource = 'com-amazon.ungraph'
    # datasource = 'com-dblp.ungraph'
    # datasource = 'com-youtube.ungraph'
    # datasource = 'com-lj.ungraph'

    graphcore = GraphCore(None, datasource)
    graph = graphcore.initGraph()
    print(graph)

    processes = list()
    for sample_rate in sample_rates:
        gc = GraphCore(graph, datasource, None, sample_rate)
        p = Process(target=samplegraphs, args=(gc, algos, samplepath,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

def sampleAll():
    """
    sample a batch
    """
    data_list = ['com-lj.ungraph']
    data_list = ['com-dblp.ungraph']
    all_sample_rates = [[0.1, 0.2, 0.3, 0.4, 0.5]]
    algos = ['SUBM']
    for datasource in data_list:
        for algo in algos:
            for rate in all_sample_rates:
                run = [datasource, [algo], rate]
                print(run)
                generate_batch_samples(run[0], run[1], run[2])

def main():
    core = 4
    if 'winston' in str(Path.home()):
        core = 48
        multiprocessing.set_start_method('spawn')
    nk.setNumberOfThreads(core)
    nk.engineering.setNumberOfThreads(core)
    start = time.time()
    sampleAll()
    end = time.time()
    print(nk.getMaxNumberOfThreads())
    print(nk.getCurrentNumberOfThreads())
    print("Time(s): {:.2f}".format(end-start))

if __name__ == '__main__':
    main()
