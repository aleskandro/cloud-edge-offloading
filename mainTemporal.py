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

rate = 1          # req/slot
time_window = 5   # min/slot
time_limit = 200  # limit time for batch execution
execution_time_scale = 2  # set me TODO

serviceProviders = PoissonRandomVariable(time_window * rate)
execution_time = ExponentialRandomVariable(execution_time_scale)

def simpleHeuristic(maxOpt, make_graph=True):
    bwOpts2 = pd.DataFrame(columns=["t", "BandwidthSaving", "Options"])
    rrOpts2 = pd.DataFrame(columns=["t", "Options", "CPU", "RAM"])
    timing = pd.DataFrame(columns=["t", "Options", "Time"])
    activeServices = pd.DataFrame(columns=["t", "Options", "Services"])
    options = UniformRandomVariable(maxOpt, maxOpt)
    generator = GeneratorForModel(servers, serviceProviders,
                                  options, containers, [cpu, ram], bandwidth, [cpuReq, ramReq], execution_time)
    generator.generate()  # TODO make multithread by not using a singleton (can I?)

    for t in range(0, time_limit, time_window):
        npp = NetworkProvider().getInstance()
        t1 = time.time()
        npp.makePlacement(maxOpt, t)
        t2 = time.time()
        # Add service providers for next time iteration
        generator.generateServiceProviders()
        timing.loc[len(timing)] = {"Options": maxOpt, "Time": t2 - t1, "t": t}
        bwOpts2.loc[len(bwOpts2)] = {"Options": maxOpt, "BandwidthSaving": npp.getBandwidthSaving(), "t": t}
        rrOpts2.loc[len(rrOpts2)] = {"Options": maxOpt, "CPU": npp.getRemainingResources()[0], "RAM": npp.getRemainingResources()[1], "t": t}
        activeServices.loc[len(activeServices)] = {"Options": maxOpt, "t": t,
                                                   "Services": len(list(
                                                       filter(lambda x: x.getDefaultOption(),
                                                              npp.getServiceProviders())))}
    if make_graph:
        makeGraph(bwOpts2, rrOpts2, timing, activeServices)
    return bwOpts2, rrOpts2, timing, activeServices

def confidenceInterval(x):
    std  = x.std()
    count= x.count()
    return 1.96*std/math.sqrt(count)


def makeGraph(bwOpts, rrOpts, timing, activeServices):
    bwOpts = bwOpts.groupby("t").agg([np.mean, confidenceInterval])
    rrOpts = rrOpts.groupby("t").agg([np.mean, confidenceInterval])
    timing = timing.groupby("t").agg([np.mean, confidenceInterval])
    activeServices["Services"] = activeServices["Services"].astype(float)
    activeServices = activeServices.groupby("t").agg([np.mean, confidenceInterval])

    fig, axs = plt.subplots(nrows=4, ncols=1)
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
    ax.errorbar(timing.index.values, timing["Time"]["mean"], yerr=timing["Time"]["confidenceInterval"], label="Time elapsed")
    ax.legend(loc="best")
    ax = axs[3]
    ax.errorbar(activeServices.index.values, activeServices["Services"]["mean"], yerr=activeServices["Services"]["confidenceInterval"], label="Active services")
    ax.legend(loc="best")
    fig.savefig("results/output.png")

def groupedHeuristic(runs, maxOpts, make_graph = True):
    bwOptss = pd.DataFrame(columns=["t", "Options", "BandwidthSaving"])
    rrOptss = pd.DataFrame(columns=["t", "Options", "CPU", "RAM"])
    timingg = pd.DataFrame(columns=["t", "Options", "Time"])
    activeServicess = pd.DataFrame(columns=["t", "Options", "Services"])
    for i in range(0, runs):
        Random.seed(i)
        bwOpts, rrOpts, timing, activeServices = simpleHeuristic(maxOpts, False)
        bwOptss = bwOptss.append(bwOpts)
        rrOptss = rrOptss.append(rrOpts)
        timingg = timingg.append(timing)
        activeServicess = activeServicess.append(activeServices)
    if make_graph:
        makeGraph(bwOptss, rrOptss, timingg, activeServicess)
    return bwOptss, rrOptss, timingg, activeServicess

os.system("rm -rf results/*")
#simple(200)
#grouped(20, 10)
#simpleHeuristic(10)
groupedHeuristic(20, 10)
