import numpy.random as Random
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import os

from Generator.Generator import *
from Generator.GeneratorBwConcave import *
from Generator.GeneratorForModel import *

from Random.NormalRandomVariable import *
from Random.UniformRandomVariable import *
from Random.ResourceDependentRandomVariable import *

Random.seed(6)
maxCpu = 30
maxRam = 30000
avgServers = 1

avgRam = maxRam/avgServers # Average ram for a single server (this is the maximum amount of ram that a container can get)
avgCpu = maxCpu/avgServers # Average cpu for a single server (this is the maximum amount of cpu that a container can get)

avgContainers = 1
avgServiceProviders = 5
# avgBandwidth = 100
K = 1.6
servers = UniformRandomVariable(avgServers, avgServers)
ram = NormalRandomVariable(avgRam, 0)
cpu = NormalRandomVariable(avgCpu, 0)

serviceProviders = UniformRandomVariable(avgServiceProviders, avgServiceProviders)
# bandwidth = NormalRandomVariable(avgBandwidth, 50)
bandwidth = ResourceDependentRandomVariable(UniformRandomVariable(1,5))
containers = UniformRandomVariable(avgContainers, avgContainers)

ramReq = UniformRandomVariable(0, K * (avgRam * avgServers) / (avgContainers * avgServiceProviders))
cpuReq = UniformRandomVariable(0, K * (avgCpu * avgServers) / (avgContainers * avgServiceProviders))


def simpleHeuristic(maxOpt):
    bwOpts = []
    rrOpts = [[], []]
    for i in range(1, maxOpt + 1):
        options = UniformRandomVariable(i, i)
        generator = GeneratorForModel(servers, serviceProviders,
                                      options, containers, [cpu, ram], bandwidth, [cpuReq, ramReq])
        generator.generate()  # TODO make multithread by not using a singleton (can I?)
        npp = NetworkProvider().getInstance()
        npp.makePlacement(i)
        bwOpts.append(npp.getBandwidthSaving())
        rrOpts[0].append(npp.getRemainingResources()[0])
        rrOpts[1].append(npp.getRemainingResources()[1])

    # TODO multiple runs error bars
    fig, axs = plt.subplots(nrows=2, ncols=1)
    ax = axs[0]
    ax.set_ylim([0, math.ceil(max(bwOpts))])
    ax.plot(range(1, len(bwOpts) + 1), bwOpts, label="Bandwidth saving")
    ax.legend(loc="best")
    ax = axs[1]
    ax.plot(range(1, len(rrOpts[0]) + 1), rrOpts[0], label="CPU")
    ax.plot(range(1, len(rrOpts[1]) + 1), rrOpts[1], label="RAM")
    ax.legend(loc="best")
    ax.set_ylim([0, 1])
    fig.savefig("results/output.png")

def simple(maxOpt):
    for i in range(1, maxOpt + 1): 
        options = UniformRandomVariable(i, i)
        # generator = Generator(servers, serviceProviders, options, containers, [ram, cpu], bandwidth, [ramReq, cpuReq])
        generator = GeneratorBwConcave(servers, serviceProviders, options, containers, [cpu, ram], bandwidth, [cpuReq, ramReq])

        generator.generate()
        os.system("glpsol --math modelglpk.mod -d scenario.dat")

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

def grouped(runs, maxOpt):
    for i in range(0, runs):
        Random.seed(i)
        simple(maxOpt)
    bwOpts = pd.read_csv("results/bandwidthByAvgOptions.csv", header=None, names=["Options","BandwidthSaving"])
    bwOpts = bwOpts.groupby("Options").agg([np.mean, confidenceInterval])
    rrOpts = pd.read_csv("results/remainingResourcesByAvgOptions.csv", header=None, names=["Options", "CPU", "RAM"])
    rrOpts = rrOpts.groupby("Options").agg([np.mean, confidenceInterval])
    fig, axs = plt.subplots(nrows=2, ncols=1)
    ax = axs[0]
    ax.set_ylim([0, math.ceil(bwOpts["BandwidthSaving"]["mean"].max())])
    ax.errorbar(bwOpts.index.values, bwOpts["BandwidthSaving"]["mean"], yerr=bwOpts["BandwidthSaving"]["confidenceInterval"], label="Bandwidth saving")
    ax.legend(loc="best")
    ax = axs[1]
    ax.errorbar(rrOpts.index.values, rrOpts["CPU"]["mean"], yerr=rrOpts["CPU"]["confidenceInterval"], label="CPU")
    ax.errorbar(rrOpts.index.values, rrOpts["RAM"]["mean"], yerr=rrOpts["RAM"]["confidenceInterval"], label="RAM")
    ax.legend(loc="best")
    ax.set_ylim([0, 1])
    fig.savefig("results/output.png")

os.system("rm -rf results/*")
#simple(200)
#grouped(20, 10)
simpleHeuristic(10)


