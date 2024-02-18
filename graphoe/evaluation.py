import networkit as nk
import networkx as nx
from littleballoffur.node_sampling import RandomNodeSampler, DegreeBasedSampler
from littleballoffur.edge_sampling import RandomEdgeSampler, RandomEdgeSamplerAdj, RandomEdgeSamplerWithPartialInduction, RandomEdgeSamplerX, HybridNodeEdgeSampler, RandomEdgeSamplerAdp
from littleballoffur.exploration_sampling import RandomWalkSampler, RandomWalkWithRestartSampler, RandomWalkWithJumpSampler, BreadthFirstSearchSampler, ForestFireSampler
import matplotlib.pyplot as plt
import math
import time
import datetime
from ordered_set import OrderedSet
import numpy as np
from metricsprocessor import MetricsProcessor
from graphcore import GraphCore
from multiprocessing import Queue, Process, set_start_method
from pathlib import Path
from picksamples import sampleset, generate_batch_samples
import os
from PIL import Image

SAMPLE_ALGO = ['X', 'PESI', 'GSES', 'NS', 'SUBM', 'RW', 'ESI', 'Y']
ALGONAME_MAP = {'True': 'True', 'X': 'DaS', 'PESI': 'PIES', 'GSES': 'GSES', 'NS': 'NS', 'SUBM': 'SubMix', 'RW': 'RW', 'ESI': 'AdjES', 'Y': 'AdapES'}
COLOR_MAP = {'True': 'red', 'X': 'gold', 'PESI': 'pink', 'GSES': 'brown', 'NS': 'green', 'SUBM': 'darkturquoise', 'RW': 'gray', 'ESI': 'darkorange', 'Y': 'blue'}
MEASURES = ['degreedist', 'coeffdist', 'kcoredist']
MEASURES_MAP = {'degreedist': 'Degree', 'coeffdist': 'Clustering Coefficient', 'kcoredist': 'K (k-core)', 'executiontime': 'Time(s)'}
GLOBAL_METRICS = ['globalclusteringcoeff', 'maxkcore', 'diameter', 'avgdegree', 'density','transitivity']
GRAPH_DATASETS = ['musae_facebook_edges',
                  'lastfm_asia_edges',
                  'deezer_europe_edges',
                  'musae_git_edges',
                  'email-Enron',
                  'loc-brightkite_edges',
                  'loc-gowalla_edges',
                  'com-amazon.ungraph',
                  'com-dblp.ungraph',
                  'com-youtube.ungraph',
                  'com-lj.ungraph'
                 ]

GRAPH_DATASETS_NAME = ['FacebookPage',
                       'LastFMAsia',
                       'DeezerEurope',
                       'GitHub',
                       'EnronEmail',
                       'Brightkite',
                       'Gowalla',
                       'AmazonProduct',
                       'DBLP',
                       'Youtube',
                       'LiveJournal'
                      ]

def constructone(queue, gc, n, xis, measure, mp):
    """
    construct sample graph y list
    """
    sampler = sampleset(gc, n)
    randomname = gc.sample_algo
    if not sampler:
        queue.put((randomname, None))
    else:
        sample = gc.construct(measure, sampler)
        if measure not in GLOBAL_METRICS:
            x, y = mp.generateY(sample.keys(), xis, sample)
        else:
            y = sample
        queue.put((randomname, y))

def constructmain(queue, gc, measure, mp):
    """
    construct original graph x,y
    """
    clusteringGraph = gc.construct(measure)
    if measure not in GLOBAL_METRICS:
        x1 = sorted(list(clusteringGraph.keys()))
        xis = mp.findSampleInterval(x1)
        x0, y0 = mp.generateY(clusteringGraph.keys(), xis, clusteringGraph)
    else:
        y0 = clusteringGraph
        xis = None
    queue.put((xis, y0))

def generateData(gc, n, measure, mp):
    """
    generate X and Ys for measure of samples by datasource
    """
    samples = SAMPLE_ALGO
    q1 = Queue()
    p1 = Process(target=constructmain, args=(q1, gc, measure, mp,))
    p1.start()
    xis, y0 = q1.get()
    p1.join()

    processes = list()
    queue = Queue()
    for randomname in samples:
        gc.setSample_algo(randomname)
        process = Process(target=constructone, args=(queue, gc, n, xis, measure, mp,))
        processes.append(process)
        process.start()
    
    # default
    time.sleep(20)

    # large data use 200
    #time.sleep(200)

    rlist = [None for i in range(len(samples)+2)]
    rlist[0] = xis
    rlist[1] = y0
    while not queue.empty():
        pair = queue.get()
        rlist[samples.index(pair[0]) + 2] = pair[1]
    for p in processes:
        p.join()

    return rlist

def drawResults(gc, n, measure, mp, xlabel, ylabel, xlog=True):
    """
    draw one fig of graph and samples for measure, write dist data
    """
    metrics_value = dict()
    samples = ['True'] + SAMPLE_ALGO
    rlist = generateData(gc, n, measure, mp)

    for i in range(1, len(rlist)):
        algo = samples[i-1]
        if algo == 'True' or algo == 'Y':
            plt.plot(rlist[0], rlist[i], linestyle='-', marker=',', label=algo)
        else:
            plt.plot(rlist[0], rlist[i], linestyle='--', marker=',', label=algo)
        metrics_value[algo] = rlist[i]

    if xlog:
        plt.xscale("log")
    plt.xlabel(xlabel)
    #plt.margins(x=0)
    plt.ylabel(ylabel)
    plt.legend()
    date = str(datetime.datetime.now()).split(" ")
    date = "{}-{}".format(date[0], date[1])
    figname = "{}{}-{}-{}-{}.png".format(gc.evaluate_data_path, gc.dataset, gc.sample_rate, measure, date)

    plt.savefig(figname)
    gc.write_eval(samples, rlist[0], metrics_value, measure)
    print("######## {} ########".format(figname))
    #plt.show()

def drawSampleResults(gc, n, measure, mp):
    """
    Draw D-stat fig of different sample ratios, write eval data
    """
    sample_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    samples = SAMPLE_ALGO
    metrics_value = dict()

    yds = list()
    for sample_rate in sample_range:
        gc.setSample_rate(sample_rate)
        lxy = generateData(gc, n, measure, mp)
        if measure not in GLOBAL_METRICS:
            yds = mp.d_stat(lxy)
        else:
            yds = lxy[2:]
        for metric_i in range(len(samples)):
            if samples[metric_i] not in metrics_value:
                metrics_value[samples[metric_i]] = list()
            metrics_value[samples[metric_i]].append(yds[metric_i])

    if measure in GLOBAL_METRICS:
        plt.plot(sample_range, [yds[1]] * len(sample_range) , linestyle='-', marker='.', label='True')
    for metric in samples:
        plt.plot(sample_range, metrics_value[metric] , linestyle='-', marker=',', label=metric)

    ylabel = measure
    plt.xlabel('Sample rate')
    if measure not in GLOBAL_METRICS:
        ylabel = '{} D-stat'.format(ylabel)
    plt.ylabel(ylabel)
    plt.legend()
    date = str(datetime.datetime.now()).split(" ")
    date = "{}-{}".format(date[0], date[1])
    figname = "{}ALL-{}-{}-{}.png".format(gc.evaluate_data_path, gc.dataset, measure, date)

    plt.savefig(figname)
    gc.write_eval(samples, sample_range, metrics_value, measure)
    print("######## {} ########".format(figname))
    #plt.show()

def draw_sample_time():
    """
    Draw run time fig of different sample ratios, write run time data
    """
    data_list = ['com-lj.ungraph']
    data_list = ['com-youtube.ungraph']
    data_list = ['lastfm_asia_edges']

    all_sample_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    algos = SAMPLE_ALGO
    measure = 'executiontime'
    parameters = []
    gc = GraphCore(None, None)
    run_samplepath = 'prundata'
    gc.set_sample_file_path(run_samplepath)
    for datasource in data_list:
        yruns = dict()
        gc.setDataset(datasource)
        for algo in algos:
            for rate in all_sample_rates:
                run = [datasource, [algo], [rate]]
                print(run)
                start = time.time()
                generate_batch_samples(run[0], run[1], run[2], run_samplepath)
                end = time.time()
                delta = end - start
                if algo not in yruns:
                    yruns[algo] = list()
                yruns[algo].append(delta)
                print("Time(s): {:.2f}".format(delta))
        gc.write_eval(algos, all_sample_rates, yruns, measure)

        for algo in algos:
            plt.plot(all_sample_rates, yruns[algo] , linestyle='-', marker=',', label=ALGONAME_MAP[algo], color=COLOR_MAP[algo])

        ylabel = measure
        plt.xlabel('Sample rate')
        plt.ylabel(ylabel)
        plt.legend()
        date = str(datetime.datetime.now()).split(" ")
        date = "{}-{}".format(date[0], date[1])
        figname = "{}ALL-{}-{}-{}.png".format(gc.evaluate_data_path, gc.dataset, measure, date)

        plt.savefig(figname)
        print("######## {} ########".format(figname))
        #plt.show()

def drawcombevaldata(gc, measures):
    """
    (DEPRACATED) draw one combined fig from eval data
    """
    evaldata = dict()
    count = len(measures)
    axs = [None for i in range(count)]
    fig, axs = plt.subplots(1, count, figsize=(16,5))
    legendfig = plt.figure()
    legends = list()
    for mi in range(count):
        measure = measures[mi]
        evaldata = gc.read_eval(measure)
        xaxis = evaldata['x']
        algos = evaldata['algos']
        ydict = evaldata['ys']
        for algo in SAMPLE_ALGO:
            line, = axs[mi].plot(xaxis, ydict[algo], linestyle='-', marker=',', linewidth=5.0, label=ALGONAME_MAP[algo], color=COLOR_MAP[algo])
            if mi == 0:
                legends.append(line)
        axs[mi].set_xlabel('Sample ratio', labelpad=0.0)
        axs[mi].set_ylabel('{} D-Statistic'.format(MEASURES_MAP[measure]))
        axs[mi].tick_params(direction='out', bottom=True, left=True, length=3)
        axs[mi].grid(False)
        axs[mi].set_title('({}) {}'.format(chr(ord('a')+mi), MEASURES_MAP[measure]), y=-0.15, pad=4.2)
        # axs[mi].text(0.1, -0.1, '({}) {}'.format(chr(ord('a')+mi),MEASURES_MAP[measure]))
        # axs[mi].get_legend().remove()
    # fig.legend(loc='upper right')
    #legendfig.legend(handles=legends, ncol=7, bbox_to_anchor=(3.1, 1.12), loc='upper right', columnspacing=0.2, prop={'size': '20'})
    legpt = legendfig.legend(handles=legends, ncol=7, columnspacing=0.2, loc='center', prop={'size': '12'})
    legpt.get_frame().set_linewidth(0.8)
    legpt.get_frame().set_edgecolor('black')
    #legendfig.savefig("legend.png", bbox_inches='tight')
    #legendfig.show()
    # axs[0].legend(handles=legends, ncol=1, bbox_to_anchor=(3.8, 1.015), loc='upper right')
    # axs[0].legend(handles=legends, ncol=1, bbox_to_anchor=(1.03, 1.12), loc='lower center', fontsize=18)
    date = str(datetime.datetime.now()).split(" ")
    date = "{}-{}".format(date[0], date[1])
    figname = "{}-{}.png".format("Dstat", gc.dataset)
    plt.show()
    #plt.savefig(figname, bbox_inches='tight')
    print("######## {} ########".format(figname))

def drawevaldata(gc, measures):
    """
    draw separate fig from eval data
    """
    evaldata = dict()
    count = len(measures)
    axs = [None for i in range(count)]
    # plt1 = plt.figure(figsize=(8,8))
    for mi in range(count):
        measure = measures[mi]
        evaldata = gc.read_eval(measure)
        xaxis = evaldata['x']
        algos = evaldata['algos']
        ydict = evaldata['ys']
        for algo in SAMPLE_ALGO:
            plt.plot(xaxis, ydict[algo], linestyle='-', marker='.', markeredgewidth=6.0, linewidth=5.0, label=ALGONAME_MAP[algo], color=COLOR_MAP[algo])
        plt.xlabel('Sampling ratio', labelpad=0.0, fontdict={'fontsize': '40'})
        plt.ylabel('D-statistic', fontdict={'fontsize': '40'})
        plt.tick_params(direction='out', bottom=True, left=True, length=7, width=4, labelsize=30)
        plt.locator_params(axis='x', tight=True, nbins=5)
        plt.locator_params(axis='y', tight=True, nbins=4)
        plt.ylim(bottom=0.0)
        plt.grid(False)
        for sp in ['top','bottom','left','right']:
            plt.gca().spines[sp].set(linewidth=2.0, color='black')
        plt.legend('', frameon=False)
        date = str(datetime.datetime.now()).split(" ")
        date = '{}-{}'.format(date[0], date[1])
        figname = '{}-{}-{}'.format('Dstat', gc.dataset, measure)
        fig_png = '{}.png'.format(figname)
        #plt.show()
        plt.savefig(fig_png, dpi=300, bbox_inches='tight')
        plt.clf()
        convert_jpeg(fig_png, '{}.jpeg'.format(figname), 'JPEG')
        #print("######## {} ########".format(figname))


def drawdistdata(gc, measures):
    """
    draw fig of dist from dist data
    """
    distdata = dict()
    samples = ['True'] + SAMPLE_ALGO
    for mi in range(len(measures)):
        measure = measures[mi]
        datasource = gc.dataset
        distdata = gc.read_eval(measure)
        xaxis = distdata['x']
        algos = distdata['algos']
        ydict = distdata['ys']
        for algo in samples:
            if algo == 'True' or algo == 'Y':
                plt.plot(xaxis, ydict[algo], linestyle='-', marker=',', label=ALGONAME_MAP[algo], color=COLOR_MAP[algo])
            else:
                plt.plot(xaxis, ydict[algo], linestyle='--', marker=',', label=ALGONAME_MAP[algo], color=COLOR_MAP[algo])
        plt.xlabel(MEASURES_MAP[measure], labelpad=0.0,fontdict={'fontsize': '15'})
        plt.ylabel('P(X<x)',fontdict={'fontsize': '15'})
        plt.tick_params(direction='out', bottom=True, left=True, length=5, width=1, labelsize=12)
        plt.legend(loc='lower right', fontsize=13, frameon=False)
        for sp in ['top','bottom','left','right']:
            plt.gca().spines[sp].set(linewidth=1.0, color='black')
        plt.grid(False)
        if measure == 'degreedist':
            plt.xscale('log')
        date = str(datetime.datetime.now()).split(" ")
        date = '{}-{}'.format(date[0], date[1])
        figname = '{}-{}-{}'.format("Dist", gc.dataset, measure)
        fig_png = '{}.png'.format(figname)
        #plt.show()
        plt.savefig(fig_png, dpi=300, bbox_inches='tight')
        plt.clf()
        convert_jpeg(fig_png, '{}.jpeg'.format(figname), 'JPEG')
        #print("######## {} ########".format(figname))

def exesample(datasource, gc, sample_rate, measures):
    """
    measures over sample data
    """
    allalgos = ['Y']
    graphcore = GraphCore(None, datasource)

    # full graph
    s1 = time.time()
    graph = graphcore.initGraph()
    print(graph)
    t1 = time.time()
    for measure in measures:
        gc.construct(measure, graph)
    e1 = time.time()
    print("------full graph time: {} {}".format(e1-s1, e1-t1))

    sample_perc = sample_rate
    n = math.ceil(graph.numberOfEdges() * sample_perc)
    print("n={}".format(n))
    gc = GraphCore(graph, datasource, None, sample_rate)

    # sample graph
    for algo in allalgos:
        s2 = time.time()
        gc.setSample_algo(algo)
        samplegraph = sampleset(gc, n)
        print(samplegraph)
        t2 = time.time()
        for measure in measures:
            gc.construct(measure, samplegraph)
        e2 = time.time()
        print("------sample graph {} time: {} {}".format(algo, (e2-s2), (e2-t2)))

def evaluateData(figtype='dstat', datasource='com-youtube.ungraph'):
    samples = SAMPLE_ALGO
    sample_rate = 0.2
    measures = MEASURES
    #datasource = 'musae_facebook_edges'
    #datasource = 'lastfm_asia_edges'
    #datasource = 'deezer_europe_edges'
    #datasource = 'musae_git_edges'

    #datasource = 'email-Enron'
    #datasource = 'loc-brightkite_edges'
    #datasource = 'loc-gowalla_edges'

    #datasource = 'com-amazon.ungraph'
    #datasource = 'com-dblp.ungraph'
    #datasource = 'com-youtube.ungraph'
    #datasource = 'com-lj.ungraph'

    gc = GraphCore(None, datasource, None, sample_rate)

    if figtype == 'exesample':
        # sample graph measures time
        exesample(datasource, gc, sample_rate, measures)
    elif figtype == 'eval_dstat':
        # draw fig from eval dstat data
        drawevaldata(gc, measures)
    elif figtype == 'eval_dist':
        # draw fig from eval dist data
        gc.use_dist()
        drawdistdata(gc, measures)
    else:
        graphcore = GraphCore(None, datasource)
        graph = graphcore.initGraph()
        print(graph)
        # drawResults(datasource, graph, 'degreecoeff', 'avg', False, 'degree', 'clustering', False)

        sample_perc = sample_rate
        n = math.ceil(graph.numberOfEdges() * sample_perc)
        print("n={}".format(n))

        gc = GraphCore(graph, datasource, None, sample_rate)
        # draw fig
        processes = list()
        for measure in measures:
            xlog = True
            gap = 10
            if measure == 'coeffdist' or measure == 'kcoredist':
                xlog = False
            gc = GraphCore(graph, datasource, None, sample_rate)
            mp = MetricsProcessor()
            if figtype == 'dist':
                gc.use_dist()
                p = Process(target=drawResults, args=(gc, n, measure, mp,  measure, 'percentage', xlog,))
            elif figtype == 'dstat':
                p = Process(target=drawSampleResults, args=(gc, n, measure, mp,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        # drawSampleResults(graph, 'degreedist', 'sum', True, 'sample ratio', 'd-stat degreedist', True, False)
        # drawSampleResults(graph, 'coeffdist', 'sum', True, 'sample ratio', 'd-stat coeffdist', True, False)

def evaluate_all(figtype='dstat'):
    """
    Evaluate a batch of datasources
    """
    data_list = ['musae_facebook_edges', 'lastfm_asia_edges', 'deezer_europe_edges', 'musae_git_edges', 'email-Enron', 'loc-brightkite_edges', 'loc-gowalla_edges', 'com-amazon.ungraph', 'com-dblp.ungraph', 'com-youtube.ungraph', 'com-lj.ungraph']
    for datasource in data_list:
        evaluateData(figtype, datasource)

def drawSampleSize():
    """
    Draw fig of relation between sample ratio and sample graph size
    """
    sample_size_file = "{}/../expdata/{}".format(os.getcwd(), "sampledatasize.txt")
    count = 0
    datasets = GRAPH_DATASETS
    datasetsname = GRAPH_DATASETS_NAME
    samplesizeY = dict()
    fullsizeY = dict()
    sampleX = [0.1, 0.2, 0.3, 0.4, 0.5]
    for ds in datasets:
        samplesizeY[ds] = [0 for i in range(len(sampleX))]
    with open(sample_size_file, "r") as fp:
        while fp:
            line = fp.readline()
            if not line:
                break
            #lastfm_asia_edges_Y_0.5_sample.txt 122089
            arr = line.split()
            if len(arr) != 2:
                continue
            if 'sample' in line:
                #lastfm_asia_edges_Y_0.5_sample.txt 122089
                for ds in samplesizeY:
                    if ds in arr[0]:
                        index_arr = arr[0].split("_")
                        if len(index_arr) < 3:
                            continue
                        if index_arr[-3] != 'Y':
                            continue
                        index_s = sampleX.index(float(index_arr[-2]))
                        if index_s < 0:
                            continue
                        samplesizeY[ds][index_s] = float(int(arr[1]))
                        break
                count += 1
            else:
                for ds in samplesizeY:
                    if ds in arr[1]:
                        fullsizeY[ds] = float(arr[0])
    for ds in samplesizeY:
        yvalues = samplesizeY[ds]
        base = fullsizeY[ds]
        for i in range(len(yvalues)):
            yvalues[i] /= base
    for ds in samplesizeY:
        ds_name = datasetsname[datasets.index(ds)]
        plt.plot(sampleX, samplesizeY[ds], linestyle='-', marker=',', label=ds_name)
    plt.xlabel('Sampling ratio', labelpad=0.0, fontdict={'fontsize': '15'})
    plt.ylabel('Sample Graph / Full Graph', fontdict={'fontsize': '15'})
    plt.tick_params(direction='out', bottom=True, left=True, length=5, width=1, labelsize=12)
    plt.legend(loc='upper left', fontsize=10, frameon=False)
    plt.xticks(np.arange(0.1,0.6,0.1))
    plt.ylim(0,0.5)
    for sp in ['top','bottom','left','right']:
        plt.gca().spines[sp].set(linewidth=1.0, color='black')
    plt.grid(False)
    date = str(datetime.datetime.now()).split(" ")
    date = '{}-{}'.format(date[0], date[1])
    figname = '{}-{}'.format("Samplesize", "Y")
    fig_png = '{}.png'.format(figname)
    #plt.show()
    plt.savefig(fig_png, dpi=300, bbox_inches='tight')
    convert_jpeg(fig_png, '{}.jpeg'.format(figname), 'JPEG')
    #print("######## {} ########".format(figname))

def drawAlphaTrace():
    """
    Draw fig of alpha trace
    """
    tracename = "log-out-com-youtube.ungraph-trace-Y-0.2-3.out"
    tracename = "log-out-com-youtube.ungraph-Y-0.2-trace-6.out"
    tracename = "log-out-com-youtube.ungraph-Y-0.2-trace-109.out"
    trace = "youtube-Y-0.2"
    alpha_trace_file = "{}/../expdata/{}".format(os.getcwd(), tracename)
    timex = list()
    degreelist = list()
    alphalist = list()
    a_sum = 0.0
    d_sum = 0.0
    limit = 1
    count = 0
    with open(alpha_trace_file, "r") as fp:
        while fp:
            line = fp.readline()
            if not line:
                break
            #5560 1688620723.154179 -6.38378239159465e-16 1.0 -0.8629072861972237
            #edge#=1195048 timestamp=1695535638.9270642 alpha=-2.0 avgdeg=5.253326066604841 degdiff=-0.0022259796197298433 randomtime=0.00011324882507324219
            tracearr = line.split()
            if len(tracearr) < 5:
                continue
            if not tracearr[1].startswith("timestamp=169") or not tracearr[0].startswith("edge#=1195048"):
                continue
            count += 1
            a_sum += float(tracearr[2].split("=")[1])
            d_sum += float(tracearr[3].split("=")[1])
            if count % limit == 0:
                timex.append(float(tracearr[1].split("=")[1]))
                alphalist.append(a_sum / count)
                degreelist.append(d_sum / count)
                a_sum = 0.0
                d_sum = 0.0
                count = 0
    startindex = 0
    tlen = len(timex)
    timex = [(timex[i] - timex[0]) for i in range(startindex, tlen)]
    degreelist = [degreelist[i] for i in range(startindex, tlen)]
    alphalist = [alphalist[i] for i in range(startindex, tlen)]
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(timex, degreelist, linestyle='-', marker=',', label='average degree', color='orangered')
    ax.set_xlabel('time(s)', fontdict={'fontsize': '23'})
    ax.set_ylabel('Average Degree', color='orangered', fontdict={'fontsize': '23'})
    ax.tick_params(direction='out', bottom=True, left=True, labelsize=18)
    ax.grid(False)
    for sp in ['top','bottom','left','right']:
        ax.spines[sp].set(linewidth=1.0, color='black')
    ax1 = ax.twinx()
    ax1.plot(timex, alphalist, linestyle='-', marker=',', label='alpha', color='slateblue')
    ax1.set_ylabel(r'$ \alpha $', rotation=0, labelpad=12.0, color='slateblue', fontdict={'fontsize': '27'})
    ax1.tick_params(direction='out', right=True, labelsize=18)
    ax1.grid(False)
    for sp in ['top','bottom','left','right']:
        ax1.spines[sp].set(linewidth=1.0, color='black')
    # plt.legend(loc='lower right', fontsize='6')
    # plt.grid(False)
    date = str(datetime.datetime.now()).split()
    date = '{}-{}'.format(date[0], date[1])
    figname = '{}-{}'.format("alphatrace", trace)
    fig_png = '{}.png'.format(figname)
    #plt.show()
    plt.savefig(fig_png, dpi=300, bbox_inches='tight')
    convert_jpeg(fig_png, '{}.jpeg'.format(figname), 'JPEG')
    #print("######## {} ########".format(figname))

def analyze(algo='Y'):
    """
    analyze dstat data saved in gc.evaluate_data_path
    """
    gc = GraphCore(None, None) 
    datasets = GRAPH_DATASETS
    datanames = GRAPH_DATASETS_NAME
    measures = MEASURES
    print("---------------- {}".format(algo))
    for measure in measures:
        print("###### {}".format(measure))
        for i in range(len(datasets)):
            data = datasets[i]
            gc.setDataset(data)
            evaldata = gc.read_eval(measure)
            algos = evaldata['algos']
            ys = evaldata['ys']
            print("{}".format(datanames[i]))
            print("   {}".format(max(ys[algo])))

def drawRuntime():
    """
    Draw run time fig of different sample ratios, write run time data
    """
    data_list = ['com-youtube.ungraph', 'com-lj.ungraph']
    #data_list = ['com-youtube.ungraph']

    all_sample_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    algos = SAMPLE_ALGO
    measure = 'executiontime'
    parameters = []
    gc = GraphCore(None, None)
    run_samplepath = 'prundata'
    gc.set_sample_file_path(run_samplepath)
    for datasource in data_list:
        gc.setDataset(datasource)
        distdata = gc.read_eval(measure)
        xaxis = distdata['x']
        algos = distdata['algos']
        ydict = distdata['ys']
        for algo in algos:
            plt.plot(xaxis, ydict[algo], linestyle='-', marker='.', markeredgewidth=4.5, linewidth=3.5, label=ALGONAME_MAP[algo], color=COLOR_MAP[algo])
        plt.xlabel('Sampling ratio', labelpad=0.0, fontdict={'fontsize': '32'})
        plt.ylabel(MEASURES_MAP[measure], fontdict={'fontsize': '32'})
        plt.tick_params(direction='out', bottom=True, left=True, length=7, width=4, labelsize=30)
        plt.locator_params(axis='x', tight=True, nbins=5)
        plt.locator_params(axis='y', tight=True, nbins=4)
        plt.ylim(bottom=0.0)
        plt.grid(False)
        for sp in ['top','bottom','left','right']:
            plt.gca().spines[sp].set(linewidth=2.0, color='black')
        plt.legend('', frameon=False)
        figname = '{}-{}'.format("Exe", datasource)
        fig_png = '{}.png'.format(figname)
        #plt.show()
        plt.savefig(fig_png, dpi=300, bbox_inches='tight') 
        plt.clf()
        convert_jpeg(fig_png, '{}.jpeg'.format(figname), 'JPEG')
        #print("######## {} ########".format(figname))

def draw_legend():
    """
    draw a fig of legend
    """
    legends = list()
    fig, axs = plt.subplots(1, 1, figsize=(19,5))
    legendfig = plt.figure()
    for algo in SAMPLE_ALGO:
        xis = [0.1*i for i in range(0,6)]
        yis = [i for i in range(0,6)]
        line, = axs.plot(xis, yis, linestyle='-', marker=',', linewidth=5.0, label=ALGONAME_MAP[algo], color=COLOR_MAP[algo])
        legends.append(line)
    #legendfig.legend(handles=legends, ncol=7, bbox_to_anchor=(3.1, 1.12), loc='upper right', columnspacing=0.2, prop={'size': '20'})
    legpt = legendfig.legend(handles=legends, ncol=8, columnspacing=0.2, loc='center', prop={'size': '12'})
    legpt.get_frame().set_linewidth(0.8)
    legpt.get_frame().set_edgecolor('black')
    figname = 'legend'
    fig_png = '{}.png'.format(figname)
    legendfig.savefig(fig_png, dpi=300, bbox_inches='tight')
    convert_jpeg(fig_png, '{}.jpeg'.format(figname), 'JPEG')
    #legendfig.show()

def convert_jpeg(oldfig, newfig, figtype):
    """
    convert fig to jpeg with dpi=300
    """
    img = Image.open(oldfig)
    rgb_img = img.convert('RGB')
    rgb_img.save(newfig, figtype, dpi=[300,300])
    print("###>> {} <<###".format(newfig))   
    try:
        os.remove(oldfig)
    except OSError:
        pass

def main():
    core = 4
    if 'winston' in str(Path.home()):
        core = 48
        set_start_method('spawn')
    nk.setNumberOfThreads(core)
    nk.engineering.setNumberOfThreads(core)
    start = time.time()
    #evaluate_all('eval_dstat')
    #evaluateData('dstat')
    #drawSampleSize()
    #drawAlphaTrace()
    analyze()
    #drawRuntime()
    #draw_legend()
    end = time.time()
    print("MaxThread#{}".format(nk.getMaxNumberOfThreads()))
    print("Thread#{}".format(nk.getCurrentNumberOfThreads()))
    print("Time(s): {:.2f}".format(end-start))

if __name__ == '__main__':
    main()
