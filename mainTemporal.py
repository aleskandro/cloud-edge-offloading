import numpy.random as Random
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from Generator.Generator import *
from Generator.GeneratorBwConcave import *
from Generator.GeneratorForModel import *

from Random.NormalRandomVariable import *
from Random.UniformRandomVariable import *
from Random.PoissonRandomVariable import *
from Random.ResourceDependentRandomVariable import *
from Random.ExponentialRandomVariable import *

Random.seed(6)
maxCpu = 30
maxRam = 30000
avgServers = 8

avgRam = maxRam/avgServers # Average ram for a single server (this is the maximum amount of ram that a container can get)
avgCpu = maxCpu/avgServers # Average cpu for a single server (this is the maximum amount of cpu that a container can get)

avgContainers = 8
avgServiceProviders = 5
# avgBandwidth = 100
K = 1.6
servers = UniformRandomVariable(avgServers, avgServers)
ram = NormalRandomVariable(avgRam, 0)
cpu = NormalRandomVariable(avgCpu, 0)

#serviceProviders = UniformRandomVariable(avgServiceProviders, avgServiceProviders)
# bandwidth = NormalRandomVariable(avgBandwidth, 50)
bandwidth = ResourceDependentRandomVariable(UniformRandomVariable(1,5))
containers = UniformRandomVariable(avgContainers, avgContainers)

ramReq = UniformRandomVariable(0, K * (avgRam * avgServers) / (avgContainers * avgServiceProviders))
cpuReq = UniformRandomVariable(0, K * (avgCpu * avgServers) / (avgContainers * avgServiceProviders))

rate = 5          # req/slot
time_window = 5   # min/slot
time_limit = 100  # limit time for batch execution
execution_time_scale = 10 # set me TODO

serviceProviders = PoissonRandomVariable(time_window * rate)
execution_time = ExponentialRandomVariable(execution_time_scale)

def simpleHeuristic(maxOpt, make_graph=True):
    bwOpts2 = pd.DataFrame(columns=["BandwidthSaving", "Options"])
    rrOpts2 = pd.DataFrame(columns=["Options", "CPU", "RAM"])
    timing = pd.DataFrame(columns=["Options", "Time"])
    for i in range(1, maxOpt + 1):
        options = UniformRandomVariable(i, i)
        generator = GeneratorForModel(servers, serviceProviders,
                                      options, containers, [cpu, ram], bandwidth, [cpuReq, ramReq])
        generator.generate()  # TODO make multithread by not using a singleton (can I?)

        for t in range(0, time_limit, time_window):
            npp = NetworkProvider().getInstance()
            t1 = time.time()
            npp.makePlacement(i, t)
            t2 = time.time()
            # Add service providers for next time iteration
            generator.generateServiceProviders()


        timing.loc[len(timing)] = {"Options": i, "Time": t2 - t1}
        bwOpts2.loc[len(bwOpts2)] = {"Options": i, "BandwidthSaving": npp.getBandwidthSaving()}
        rrOpts2.loc[len(rrOpts2)] = {"Options": i, "CPU": npp.getRemainingResources()[0], "RAM": npp.getRemainingResources()[1]}
    if make_graph:
        makeGraph(bwOpts2, rrOpts2, timing)
    return bwOpts2, rrOpts2, timing

def simple(maxOpt):
    timing = pd.DataFrame(columns=["Options", "Time"])
    for i in range(1, maxOpt + 1):
        options = UniformRandomVariable(i, i)
        # generator = Generator(servers, serviceProviders, options, containers, [ram, cpu], bandwidth, [ramReq, cpuReq])
        generator = GeneratorBwConcave(servers, serviceProviders, options, containers, [cpu, ram], bandwidth,
                                       [cpuReq, ramReq], execution_time)

        generator.generate()
        t1 = time.time()
        os.system("glpsol --math modelglpk.mod -d scenario.dat")
        t2 = time.time()
        timing.loc[len(timing)] = {"Options": i, "Time": t2-t1}
    return timing

def confidenceInterval(x):
    std  = x.std()
    count= x.count()
    return 1.96*std/math.sqrt(count)


def confidenceIntervalMax(x):
    mean = x.mean()
    std  = x.std()
    count= x.count()
    return mean + 1.96*std/math.sqrt(count)

def confidenceIntervalMin(x):
    mean = x.mean()
    std  = x.std()
    count= x.count()
    return mean - 1.96*std/math.sqrt(count)

def makeGraph(bwOpts, rrOpts, timing):
    bwOpts = bwOpts.groupby("Options").agg([np.mean, confidenceInterval])
    rrOpts = rrOpts.groupby("Options").agg([np.mean, confidenceInterval])
    timing = timing.groupby("Options").agg([np.mean, confidenceInterval])

    fig, axs = plt.subplots(nrows=3, ncols=1)
    ax = axs[0]
    ax.set_ylim([0, math.ceil(bwOpts["BandwidthSaving"]["mean"].max())])
    ax.errorbar(bwOpts.index.values, bwOpts["BandwidthSaving"]["mean"], yerr=bwOpts["BandwidthSaving"]["confidenceInterval"], label="Bandwidth saving")
    ax.legend(loc="best")
    ax = axs[1]
    ax.errorbar(rrOpts.index.values, rrOpts["CPU"]["mean"], yerr=rrOpts["CPU"]["confidenceInterval"], label="CPU")
    ax.errorbar(rrOpts.index.values, rrOpts["RAM"]["mean"], yerr=rrOpts["RAM"]["confidenceInterval"], label="RAM")
    ax.legend(loc="best")
    ax.set_ylim([0, 1])
    ax = axs[2]
    ax.errorbar(timing.index.values, timing["Time"]["mean"], yerr=timing["Time"]["mean"], label="Time elapsed")
    ax.legend(loc="best")
    fig.savefig("results/output.png")

def makeGraphTogether(bwOptsILP, rrOptsILP, timingILP, bwOptsH, rrOptsH, timingH):
    bwOptsILP = bwOptsILP.groupby("Options").agg([np.mean, confidenceInterval])
    rrOptsILP = rrOptsILP.groupby("Options").agg([np.mean, confidenceInterval])
    bwOptsH["BandwidthSaving"] = bwOptsH["BandwidthSaving"].astype(float)
    bwOptsH = bwOptsH.groupby("Options").agg([np.mean, confidenceInterval])
    rrOptsH = rrOptsH.groupby("Options").agg([np.mean, confidenceInterval])
    timingILP = timingILP.groupby("Options").agg([np.mean, confidenceInterval])
    timingH = timingH.groupby("Options").agg([np.mean, confidenceInterval])
    fig, axs = plt.subplots(nrows=3, ncols=1)
    ax = axs[0]
    ax.set_ylim([0, avgServiceProviders])  # math.ceil(bwOptsILP["BandwidthSaving"]["mean"].max())])
    ax.errorbar(bwOptsILP.index.values, bwOptsILP["BandwidthSaving"]["mean"], yerr=bwOptsILP["BandwidthSaving"]["confidenceInterval"], label="Bandwidth saving (optimal)")
    ax.errorbar(bwOptsH.index.values, bwOptsH["BandwidthSaving"]["mean"], yerr=bwOptsH["BandwidthSaving"]["confidenceInterval"], label="Bandwidth saving (heuristic)")
    ax.legend(loc="best")
    ax = axs[1]
    ax.errorbar(rrOptsILP.index.values, rrOptsILP["CPU"]["mean"], yerr=rrOptsILP["CPU"]["confidenceInterval"], label="CPU (optimal)")
    ax.errorbar(rrOptsH.index.values, rrOptsH["CPU"]["mean"], yerr=rrOptsH["CPU"]["confidenceInterval"], label="CPU (heuristic)")
    ax.errorbar(rrOptsILP.index.values, rrOptsILP["RAM"]["mean"], yerr=rrOptsILP["RAM"]["confidenceInterval"], label="RAM (optimal)")
    ax.errorbar(rrOptsH.index.values, rrOptsH["RAM"]["mean"], yerr=rrOptsH["RAM"]["confidenceInterval"], label="RAM (heuristic)")
    ax.legend(loc="best")
    ax.set_ylim([0, 1])
    ax = axs[2]
    ax.errorbar(timingILP.index.values, timingILP["Time"]["mean"], yerr=timingILP["Time"]["confidenceInterval"], label="Time elapsed (optimal)")
    ax.errorbar(timingH.index.values, timingH["Time"]["mean"], yerr=timingH["Time"]["confidenceInterval"], label="Time elapsed (heuristic)")
    ax.legend(loc="best")
    fig.savefig("results/output.png")

def grouped(runs, maxOpt, make_graph = True):
    timingg = pd.DataFrame(columns=["Options", "Time"])
    for i in range(0, runs):
        Random.seed(i)
        timing = simple(maxOpt)
        timingg = timingg.append(timing)
    bwOpts = pd.read_csv("results/bandwidthByAvgOptions.csv", header=None, names=["Options","BandwidthSaving"])
    rrOpts = pd.read_csv("results/remainingResourcesByAvgOptions.csv", header=None, names=["Options", "CPU", "RAM"])
    if make_graph:
        makeGraph(bwOpts, rrOpts, timingg)
    return bwOpts, rrOpts, timingg

def groupedHeuristic(runs, maxOpts, make_graph = True):
    bwOptss = pd.DataFrame(columns=["Options", "BandwidthSaving"])
    rrOptss = pd.DataFrame(columns=["Options", "CPU", "RAM"])
    timingg = pd.DataFrame(columns=["Options", "Time"])
    for i in range(0, runs):
        Random.seed(i)
        bwOpts, rrOpts, timing = simpleHeuristic(maxOpts, False)
        bwOptss = bwOptss.append(bwOpts)
        rrOptss = rrOptss.append(rrOpts)
        timingg = timing.append(timing)
    if make_graph:
        makeGraph(bwOptss, rrOptss, timingg)
    return bwOptss, rrOptss, timingg

def groupedTogether(runs, maxOpts):
    bwOptsILP, rrOptsILP, timingILP = grouped(runs, maxOpts, False)
    bwOptsH, rrOptsH, timingH = groupedHeuristic(runs, maxOpts, False)
    makeGraphTogether(bwOptsILP, rrOptsILP, timingILP, bwOptsH, rrOptsH, timingH)

os.system("rm -rf results/*")
#simple(200)
#grouped(20, 10)
#simpleHeuristic(10)
groupedTogether(20, 10)

