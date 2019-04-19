import numpy.random as Random
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import glob
import re
from cycler import cycler

from Generator.Generator import *
from Generator.GeneratorBwConcave import *
from Generator.GeneratorForModel import *

from Random.NormalRandomVariable import *
from Random.UniformRandomVariable import *
from Random.ResourceDependentRandomVariable import *


def generate_input_datas(maxCpu=30, maxRam=30000, avgServers=8, avgContainers=8, avgServiceProviders=5, K=1.6):

    avgRam = maxRam/avgServers
    avgCpu = maxCpu/avgServers
    # avgBandwidth = 100
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq

    servers = UniformRandomVariable(avgServers, avgServers)
    ram = NormalRandomVariable(avgRam, 0)
    cpu = NormalRandomVariable(avgCpu, 0)

    serviceProviders = UniformRandomVariable(avgServiceProviders, avgServiceProviders)
    # bandwidth = NormalRandomVariable(avgBandwidth, 50)
    bandwidth = ResourceDependentRandomVariable(UniformRandomVariable(1,5))
    containers = UniformRandomVariable(avgContainers, avgContainers)

    ramReq = UniformRandomVariable(0, K * (avgRam * avgServers) / (avgContainers * avgServiceProviders))
    cpuReq = UniformRandomVariable(0, K * (avgCpu * avgServers) / (avgContainers * avgServiceProviders))
    return servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq

def simpleHeuristic(maxOpt, make_graph=True):
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq
    bwOpts2 = pd.DataFrame(columns=["BandwidthSaving", "Options"])
    rrOpts2 = pd.DataFrame(columns=["Options", "CPU", "RAM"])
    timing = pd.DataFrame(columns=["Options", "Time"])
    for i in range(1, maxOpt + 1):
        options = UniformRandomVariable(i, i)
        generator = GeneratorForModel(servers, serviceProviders,
                                      options, containers, [cpu, ram], bandwidth, [cpuReq, ramReq])
        generator.generate()  # TODO make multithread by not using a singleton (can I?)
        npp = NetworkProvider().getInstance()
        t1 = time.time()
        npp.makePlacement(i)
        t2 = time.time()
        timing.loc[len(timing)] = {"Options": i, "Time": t2 - t1}
        bwOpts2.loc[len(bwOpts2)] = {"Options": i, "BandwidthSaving": npp.getBandwidthSaving()}
        rrOpts2.loc[len(rrOpts2)] = {"Options": i, "CPU": npp.getRemainingResources()[0], "RAM": npp.getRemainingResources()[1]}
    if make_graph:
        makeGraph(bwOpts2, rrOpts2, timing)
    return bwOpts2, rrOpts2, timing

def groupedHeuristic(runs, maxOpts, make_graph = True):
    bwOptss = pd.DataFrame(columns=["Options", "BandwidthSaving"])
    rrOptss = pd.DataFrame(columns=["Options", "CPU", "RAM"])
    timingg = pd.DataFrame(columns=["Options", "Time"])
    for i in range(0, runs):
        Random.seed(i)
        bwOpts, rrOpts, timing = simpleHeuristic(maxOpts, False)
        bwOptss = bwOptss.append(bwOpts)
        rrOptss = rrOptss.append(rrOpts)
        timingg = timingg.append(timing)
    if make_graph:
        makeGraph(bwOptss, rrOptss, timingg)
    return bwOptss, rrOptss, timingg

def simple(maxOpt):
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq
    timing = pd.DataFrame(columns=["Options", "Time"])
    for i in range(1, maxOpt + 1): 
        options = UniformRandomVariable(i, i)
        # generator = Generator(servers, serviceProviders, options, containers, [ram, cpu], bandwidth, [ramReq, cpuReq])
        generator = GeneratorBwConcave(servers, serviceProviders, options, containers, [cpu, ram], bandwidth, [cpuReq, ramReq])

        generator.generate()
        t1 = time.time()
        os.system("timeout 10 glpsol --math modelglpk.mod -d scenario.dat")
        t2 = time.time()
        timing.loc[len(timing)] = {"Options": i, "Time": t2-t1}
    return timing

def grouped(runs, maxOpt, make_graph = True):
    timingg = pd.DataFrame(columns=["Options", "Time"])
    for i in range(0, runs):
        Random.seed(i)
        timing = simple(maxOpt)
        timingg = timingg.append(timing)
    bwOpts = pd.read_csv("tresults/bandwidthByAvgOptions.csv", header=None, names=["Options","BandwidthSaving"])
    rrOpts = pd.read_csv("tresults/remainingResourcesByAvgOptions.csv", header=None, names=["Options", "CPU", "RAM"])
    if make_graph:
        makeGraph(bwOpts, rrOpts, timingg)
    return bwOpts, rrOpts, timingg

def groupedTogether(runs, maxOpts):
    bwOptsILP, rrOptsILP, timingILP = grouped(runs, maxOpts, False)
    bwOptsH, rrOptsH, timingH = groupedHeuristic(runs, maxOpts, False)
    makeGraphTogether(bwOptsILP, rrOptsILP, timingILP, bwOptsH, rrOptsH, timingH)

def groupedTogetherSaveDifferentServersFixedCC(runs, maxOpts, sList):
    for avgServer in sList:
        generate_input_datas(avgServers=avgServer, avgContainers=1)
        filename = "cmpILPH-Options-VaryingServers_%d_Servers" % avgServer
        if len(glob.glob("results/%s*" % filename)) == 0:
            bwOptsILP, rrOptsILP, timingILP = grouped(runs, maxOpts, False)
            bwOptsH, rrOptsH, timingH = groupedHeuristic(runs, maxOpts, False)
            save_to_file(filename, bwOptsILP, rrOptsILP, timingILP, bwOptsH, rrOptsH, timingH)

def groupedTogetherSaveFixedServersDiferentCC(runs, maxOpts, ccList):
    for avgContainer in ccList:
        generate_input_datas(avgServers=8, avgContainers=avgContainer)
        filename = "cmpILPH-Options-VaryingContainers_%d_Containers" % avgContainer
        os.system("rm -rf tresults/*")
        if len(glob.glob("results/%s*" % filename)) == 0:
            bwOptsILP, rrOptsILP, timingILP = grouped(runs, maxOpts, False)
            bwOptsH, rrOptsH, timingH = groupedHeuristic(runs, maxOpts, False)
            save_to_file(filename, bwOptsILP, rrOptsILP, timingILP, bwOptsH, rrOptsH, timingH)

def save_to_file(filename, bwOptsILP, rrOptsILP, timingILP, bwOptsH, rrOptsH, timingH):
    bwOptsILP.to_csv("results/%s-bwOptsILP.csv" % filename)
    rrOptsILP.to_csv("results/%s-rrOptsILP.csv" % filename)
    timingILP.to_csv("results/%s-timingILP.csv" % filename)
    bwOptsH.to_csv("results/%s-bwOptsH.csv" % filename)
    rrOptsH.to_csv("results/%s-rrOptsH.csv" % filename)
    timingH.to_csv("results/%s-timingH.csv" % filename)


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
    ax.errorbar(timing.index.values, timing["Time"]["mean"], yerr=timing["Time"]["confidenceInterval"], label="Time elapsed")
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
    ax.set_ylim([0, 5]) #avgServiceProviders])  # math.ceil(bwOptsILP["BandwidthSaving"]["mean"].max())])
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

def make_graph_from_file(filename, varied):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15,25))
    monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':']) * cycler('marker',['^', ',', '.']))
    for ax in axs:
        ax.set_prop_cycle(monochrome)

    ax = axs[0]
    ax.set_ylim([0, 5]) # avgServiceProviders])  # math.ceil(bwOptsILP["BandwidthSaving"]["mean"].max())])

    for file in glob.glob("results/" + filename + "*-bwOptsILP.csv"):
        bwOptsILP = pd.read_csv(file)
        bwOptsILP["BandwidthSaving"] = bwOptsILP["BandwidthSaving"].astype(float)
        bwOptsILP = bwOptsILP.groupby("Options").agg([np.mean, confidenceInterval])
        ax.errorbar(bwOptsILP.index.values, bwOptsILP["BandwidthSaving"]["mean"],
                    yerr=bwOptsILP["BandwidthSaving"]["confidenceInterval"],
                    label="Bandwidth saving (optimal, %s %s)" % (re.findall("_(.*)_", file)[0], varied))

    for file in glob.glob("results/" + filename + "*-bwOptsH.csv"):
        bwOptsH = pd.read_csv(file)
        bwOptsH["BandwidthSaving"] = bwOptsH["BandwidthSaving"].astype(float)
        bwOptsH = bwOptsH.groupby("Options").agg([np.mean, confidenceInterval])
        ax.errorbar(bwOptsH.index.values, bwOptsH["BandwidthSaving"]["mean"],
                    yerr=bwOptsH["BandwidthSaving"]["confidenceInterval"],
                    label="Bandwidth saving (heuristic, %s %s)" % (re.findall("_(.*)_", file)[0], varied))

    ax = axs[1]
    ax.set_ylim([0, 1])

    for file in glob.glob("results/" + filename + "*-rrOptsILP.csv"):
        rrOptsILP = pd.read_csv(file)
        rrOptsILP = rrOptsILP.groupby("Options").agg([np.mean, confidenceInterval])
        ax.errorbar(rrOptsILP.index.values, rrOptsILP["CPU"]["mean"],
                    yerr=rrOptsILP["CPU"]["confidenceInterval"],
                    label="CPU (optimal, %s %s)" % (re.findall("_(.*)_", file)[0], varied))
        ax.errorbar(rrOptsILP.index.values, rrOptsILP["RAM"]["mean"],
                    yerr=rrOptsILP["RAM"]["confidenceInterval"],
                    label="RAM (optimal, %s %s)" % (re.findall("_(.*)_", file)[0], varied))

    for file in glob.glob("results/" + filename + "*-rrOptsH.csv"):
        rrOptsH = pd.read_csv(file)
        rrOptsH = rrOptsH.groupby("Options").agg([np.mean, confidenceInterval])
        ax.errorbar(rrOptsH.index.values, rrOptsH["CPU"]["mean"],
                    yerr=rrOptsH["CPU"]["confidenceInterval"],
                    label="CPU (optimal, %s %s)" % (re.findall("_(.*)_", file)[0], varied))
        ax.errorbar(rrOptsH.index.values, rrOptsH["RAM"]["mean"],
                    yerr=rrOptsH["RAM"]["confidenceInterval"],
                    label="RAM (heuristic, %s %s)" % (re.findall("_(.*)_", file)[0], varied))

    ax = axs[2]
    for file in glob.glob("results/" + filename + "*-timingILP.csv"):
        timingILP = pd.read_csv(file)
        timingILP = timingILP.groupby("Options").agg([np.mean, confidenceInterval])
        ax.errorbar(timingILP.index.values, timingILP["Time"]["mean"],
                    yerr=timingILP["Time"]["confidenceInterval"],
                    label="Time elapsed (optimal, %s %s)" %  (re.findall("_(.*)_", file)[0], varied))

    for file in glob.glob("results/" + filename + "*-timingH.csv"):
        timingH = pd.read_csv(file)
        timingH = timingH.groupby("Options").agg([np.mean, confidenceInterval])
        ax.errorbar(timingH.index.values, timingH["Time"]["mean"],
                    yerr=timingH["Time"]["confidenceInterval"],
                    label="Time elapsed (heuristic, %s %s)" %  (re.findall("_(.*)_", file)[0], varied))
    for ax in axs:
        ax.legend(loc="best")
        ax.set_xlabel("Number of options per service provider")

    fig.savefig("results/%s.png" % filename)

def confidenceInterval(x):
    std = x.std()
    count = x.count()
    return 1.96*std/math.sqrt(count)


os.system("rm -rf tresults/*")
#generate_input_datas()
#simple(200)
#grouped(20, 10)
#simpleHeuristic(10)
#groupedTogether(20, 10)

groupedTogetherSaveDifferentServersFixedCC(20, 10, [1, 2, 4, 8])
groupedTogetherSaveFixedServersDiferentCC(20, 10, [1, 2, 4, 8])
make_graph_from_file("cmpILPH-Options-VaryingContainers", "containers")
make_graph_from_file("cmpILPH-Options-VaryingServers", "servers")
