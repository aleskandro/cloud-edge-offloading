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

maxxCpu = 30
maxxRam = 30000
defaultOptions = 5
def generate_input_datas(avgCpu=32, avgRam=32768, avgServers=8, avgContainers=8, avgServiceProviders=50, K=1.5):
    global maxxCpu, maxxRam

    maxxCpu = avgCpu * avgServers
    maxxRam = avgRam * avgServers
    print(maxxCpu, maxxRam)
    # avgBandwidth = 100
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq

    servers = UniformRandomVariable(avgServers, avgServers)
    ram = NormalRandomVariable(avgRam, 0)
    cpu = NormalRandomVariable(avgCpu, 0)

    serviceProviders = UniformRandomVariable(avgServiceProviders, avgServiceProviders)
    # bandwidth = NormalRandomVariable(avgBandwidth, 50)
    bandwidth = ResourceDependentRandomVariable(UniformRandomVariable(1,5))
    containers = UniformRandomVariable(avgContainers, avgContainers)

    ramReq = UniformRandomVariable(0, K * (avgRam * avgServers) / (avgContainers * avgServiceProviders), False)
    cpuReq = UniformRandomVariable(0, K * (avgCpu * avgServers) / (avgContainers * avgServiceProviders), False)
    return servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq

def simpleHeuristic(maxOpt, make_graph=True, placement_func=NetworkProvider().getInstance().makePlacement):
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq
    bwOpts2 = pd.DataFrame(columns=["BandwidthSaving", "Options"])
    rrOpts2 = pd.DataFrame(columns=["Options", "CPU", "RAM"])
    timing = pd.DataFrame(columns=["Options", "Time"])
    i = 1
    while (i <= maxOpt + 1):
        options = UniformRandomVariable(i, i)
        generate_input_datas()
        generator = GeneratorForModel(servers, serviceProviders,
                                      options, containers, [cpu, ram], bandwidth, [cpuReq, ramReq])
        generator.generate()  # TODO make multithread by not using a singleton (can I?)
        npp = NetworkProvider().getInstance()
        t1 = time.time()
        placement_func(i)
        t2 = time.time()
        timing.loc[len(timing)] = {"Options": i, "Time": t2 - t1}
        bwOpts2.loc[len(bwOpts2)] = {"Options": i, "BandwidthSaving": npp.getBandwidthSaving()}
        rrOpts2.loc[len(rrOpts2)] = {"Options": i, "CPU": npp.getRemainingResources()[0], "RAM": npp.getRemainingResources()[1]}
        i *= 2
    if make_graph:
        makeGraph(bwOpts2, rrOpts2, timing)
    return bwOpts2, rrOpts2, timing

def simpleHeuristicVarServers(maxServers, make_graph=True):
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq
    bwOpts2 = pd.DataFrame(columns=["BandwidthSaving", "Options"])
    rrOpts2 = pd.DataFrame(columns=["Options", "CPU", "RAM"])
    timing = pd.DataFrame(columns=["Options", "Time"])
    i = 1
    while (i <= maxServers + 1):
        options = UniformRandomVariable(defaultOptions, defaultOptions)
        generate_input_datas(avgServers=i)
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
        i *= 2
    if make_graph:
        makeGraph(bwOpts2, rrOpts2, timing)
    return bwOpts2, rrOpts2, timing

def simpleHeuristicVarContainers(maxContainers, make_graph=True):
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq
    bwOpts2 = pd.DataFrame(columns=["BandwidthSaving", "Options"])
    rrOpts2 = pd.DataFrame(columns=["Options", "CPU", "RAM"])
    timing = pd.DataFrame(columns=["Options", "Time"])
    i = 1
    while (i <= maxContainers + 1):
        options = UniformRandomVariable(defaultOptions, defaultOptions)
        generate_input_datas(avgContainers=i)
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
        i *= 2
    if make_graph:
        makeGraph(bwOpts2, rrOpts2, timing)
    return bwOpts2, rrOpts2, timing

def groupedHeuristic(runs, maxOpts, function_to_call, make_graph = True):
    bwOptss = pd.DataFrame(columns=["Options", "BandwidthSaving"])
    rrOptss = pd.DataFrame(columns=["Options", "CPU", "RAM"])
    timingg = pd.DataFrame(columns=["Options", "Time"])
    for i in range(0, runs):
        Random.seed(i)
        bwOpts, rrOpts, timing = function_to_call(maxOpts, False)
        bwOptss = bwOptss.append(bwOpts)
        rrOptss = rrOptss.append(rrOpts)
        timingg = timingg.append(timing)
    if make_graph:
        makeGraph(bwOptss, rrOptss, timingg)
    return bwOptss, rrOptss, timingg

def groupedHeuristic2(runs, maxOpts, function_to_call, make_graph = True, func_placement=NetworkProvider().getInstance().makePlacement):
    bwOptss = pd.DataFrame(columns=["Options", "BandwidthSaving"])
    rrOptss = pd.DataFrame(columns=["Options", "CPU", "RAM"])
    timingg = pd.DataFrame(columns=["Options", "Time"])
    for i in range(0, runs):
        Random.seed(i)
        bwOpts, rrOpts, timing = function_to_call(maxOpts, False, func_placement)
        bwOptss = bwOptss.append(bwOpts)
        rrOptss = rrOptss.append(rrOpts)
        timingg = timingg.append(timing)
    if make_graph:
        makeGraph(bwOptss, rrOptss, timingg)
    return bwOptss, rrOptss, timingg

def simple(maxOpt):
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq
    timing = pd.DataFrame(columns=["Options", "Time"])
    i = 1
    while(i <= maxOpt + 1):
        options = UniformRandomVariable(i, i)
        # generator = Generator(servers, serviceProviders, options, containers, [ram, cpu], bandwidth, [ramReq, cpuReq])
        generator = GeneratorBwConcave(servers, serviceProviders, options, containers, [cpu, ram], bandwidth, [cpuReq, ramReq])

        generator.generate()
        t1 = time.time()
        os.system("glpsol --math modelglpk.mod -d scenario.dat --tmlim 600")
        t2 = time.time()
        timing.loc[len(timing)] = {"Options": i, "Time": t2-t1}
        i *= 2
    return timing

def simpleVarServers(maxOpt):
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq
    timing = pd.DataFrame(columns=["Options", "Time"])
    i = 1
    while(i <= maxOpt + 1):
        options = UniformRandomVariable(defaultOptions, defaultOptions)
        generate_input_datas(avgServers=i)
        # generator = Generator(servers, serviceProviders, options, containers, [ram, cpu], bandwidth, [ramReq, cpuReq])
        generator = GeneratorBwConcave(servers, serviceProviders, options, containers, [cpu, ram], bandwidth, [cpuReq, ramReq])

        generator.generate()
        t1 = time.time()
        os.system("glpsol --math modelglpk.mod -d scenario.dat --tmlim 600")
        t2 = time.time()
        timing.loc[len(timing)] = {"Options": i, "Time": t2-t1}
        i *= 2
    return timing

def simpleVarContainers(maxOpt):
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq
    timing = pd.DataFrame(columns=["Options", "Time"])
    i = 1
    while(i <= maxOpt + 1):
        options = UniformRandomVariable(defaultOptions, defaultOptions)
        generate_input_datas(avgContainers=i)
        # generator = Generator(servers, serviceProviders, options, containers, [ram, cpu], bandwidth, [ramReq, cpuReq])
        generator = GeneratorBwConcave(servers, serviceProviders, options, containers, [cpu, ram], bandwidth, [cpuReq, ramReq])

        generator.generate()
        t1 = time.time()
        os.system("timeout 600 glpsol --math modelglpk.mod -d scenario.dat --tmlim 580")
        t2 = time.time()
        timing.loc[len(timing)] = {"Options": i, "Time": t2-t1}
        i *= 2
    return timing

def grouped(runs, maxOpt, function_to_call, make_graph = True):
    timingg = pd.DataFrame(columns=["Options", "Time"])
    for i in range(0, runs):
        Random.seed(i)
        timing = function_to_call(maxOpt)
        timingg = timingg.append(timing)
    bwOpts = pd.read_csv("tresults/bandwidthByAvgOptions.csv", header=None, names=["Servers", "Containers","Options","BandwidthSaving"])
    rrOpts = pd.read_csv("tresults/remainingResourcesByAvgOptions.csv", header=None, names=["Servers", "Containers", "Options", "CPU", "RAM"])
    if make_graph:
        makeGraph(bwOpts, rrOpts, timingg)
    return bwOpts, rrOpts, timingg

def groupedTogether(runs, maxOpts):
    bwOptsILP, rrOptsILP, timingILP = grouped(runs, maxOpts, simple, False)
    bwOptsH, rrOptsH, timingH = groupedHeuristic(runs, maxOpts, simpleHeuristic, False)
    makeGraphTogether(bwOptsILP, rrOptsILP, timingILP, bwOptsH, rrOptsH, timingH)

def groupedTogetherSaveVarOptionsFixedServersFixedCC(runs, maxOpts):
    filename = "cmpILPH-Options"
    os.system("rm -rf tresults/*")
    if len(glob.glob("results/%s*" % filename)) == 0:
        bwOptsILP, rrOptsILP, timingILP = grouped(runs, maxOpts, simple, False)
        bwOptsH, rrOptsH, timingH = groupedHeuristic(runs, maxOpts, simpleHeuristic, False)
        save_to_file(filename, bwOptsILP, rrOptsILP, timingILP, bwOptsH, rrOptsH, timingH)

def groupedTogetherSaveFixedOptionsVarServersFixedCC(runs, maxOpts):
    filename = "cmpILPH-Servers"
    os.system("rm -rf tresults/*")
    if len(glob.glob("results/%s*" % filename)) == 0:
        bwOptsILP, rrOptsILP, timingILP = grouped(runs, maxOpts, simpleVarServers, False)
        bwOptsH, rrOptsH, timingH = groupedHeuristic(runs, maxOpts, simpleHeuristicVarServers, False)
        save_to_file(filename, bwOptsILP, rrOptsILP, timingILP, bwOptsH, rrOptsH, timingH)

def groupedTogetherSaveFixedOptionsFixedServersVarCC(runs, maxOpts):
    filename = "cmpILPH-Containers"
    os.system("rm -rf tresults/*")
    if len(glob.glob("results/%s*" % filename)) == 0:
        bwOptsILP, rrOptsILP, timingILP = grouped(runs, maxOpts, simpleVarContainers, False)
        bwOptsH, rrOptsH, timingH = groupedHeuristic(runs, maxOpts, simpleHeuristicVarContainers, False)
        save_to_file(filename, bwOptsILP, rrOptsILP, timingILP, bwOptsH, rrOptsH, timingH)

def naiveHeuristicSaveVarOptions(runs, maxOpts):
    filename = "cmpHN-Options"
    os.system("rm -rf tresults/*")
    if len(glob.glob("results/%s*" % filename)) == 0:
        bwOptsILP, rrOptsILP, timingILP = groupedHeuristic2(runs, maxOpts, simpleHeuristic, False, NetworkProvider().getInstance().make_placement_naive)  # Read naive
        bwOptsH, rrOptsH, timingH = groupedHeuristic2(runs, maxOpts, simpleHeuristic, False)
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

def make_graph_from_file(filename, ilp_key, xlabel, ilp_label="optimal", h_label="heuristic"):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10,10))
    monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':']) * cycler('marker',['^', ',', '.']))
    #for ax in axs:
    #    ax.set_prop_cycle(monochrome)

    ax = axs[0]
    max_y = 0
    ax.set_ylim([0, max_y])
    ax.set_ylabel("Utility")
    for file in glob.glob("results/" + filename + "*-bwOptsILP.csv"):
        bwOptsILP = pd.read_csv(file)
        bwOptsILP["BandwidthSaving"] = bwOptsILP["BandwidthSaving"].astype(float)
        bwOptsILP = bwOptsILP.groupby(ilp_key).agg([np.mean, confidenceInterval])
        ax.errorbar(bwOptsILP.index.values, bwOptsILP["BandwidthSaving"]["mean"],
                    yerr=bwOptsILP["BandwidthSaving"]["confidenceInterval"],
                    label="Bandwidth saving (%s)" % ilp_label)
        max_y = max(math.ceil(bwOptsILP["BandwidthSaving"]["mean"].max()) + 2, max_y)
        ax.set_ylim([0, max_y])
    for file in glob.glob("results/" + filename + "*-bwOptsH.csv"):
        bwOptsH = pd.read_csv(file)
        bwOptsH["BandwidthSaving"] = bwOptsH["BandwidthSaving"].astype(float)
        bwOptsH = bwOptsH.groupby("Options").agg([np.mean, confidenceInterval])
        ax.errorbar(bwOptsH.index.values, bwOptsH["BandwidthSaving"]["mean"],
                    yerr=bwOptsH["BandwidthSaving"]["confidenceInterval"],
                    label="Bandwidth saving (%s)" % h_label)
        max_y = max(math.ceil(bwOptsH["BandwidthSaving"]["mean"].max()) + 2, max_y)
        ax.set_ylim([0, max_y])
    ax = axs[1]
    ax.set_ylim([0, 100])
    ax.set_ylabel("Available resources after placement (%)")
    width = 0.35
    for file in glob.glob("results/" + filename + "*-rrOptsILP.csv"):
        rrOptsILP = pd.read_csv(file)
        rrOptsILP = rrOptsILP.groupby(ilp_key).agg([np.mean, confidenceInterval])
        ax.bar(rrOptsILP.index.values - 3*width/8, rrOptsILP["CPU"]["mean"]*100, width/4,
                    yerr=rrOptsILP["CPU"]["confidenceInterval"]*100,
                    label="CPU (%s)" % ilp_label)
        ax.bar(rrOptsILP.index.values - 1*width/8, rrOptsILP["RAM"]["mean"]*100, width/4,
                    yerr=rrOptsILP["RAM"]["confidenceInterval"]*100,
                    label="RAM (%s)" % ilp_label)

    for file in glob.glob("results/" + filename + "*-rrOptsH.csv"):
        rrOptsH = pd.read_csv(file)
        rrOptsH = rrOptsH.groupby("Options").agg([np.mean, confidenceInterval])
        ax.bar(rrOptsH.index.values + 1*width/8, rrOptsH["CPU"]["mean"]*100, width/4,
                    yerr=rrOptsH["CPU"]["confidenceInterval"]*100,
                    label="CPU (%s)" % h_label)
        ax.bar(rrOptsH.index.values + 3*width/8, rrOptsH["RAM"]["mean"]*100, width/4,
                    yerr=rrOptsH["RAM"]["confidenceInterval"]*100,
                    label="RAM (%s)" % h_label)

    ax = axs[2]
    ax.set_ylabel("Time elapsed (s)")
    for file in glob.glob("results/" + filename + "*-timingILP.csv"):
        timingILP = pd.read_csv(file)
        timingILP = timingILP.groupby("Options").agg([np.mean, confidenceInterval])
        ax.errorbar(timingILP.index.values, timingILP["Time"]["mean"],
                    yerr=timingILP["Time"]["confidenceInterval"],
                    label="Time elapsed (%s)" % ilp_label)

    for file in glob.glob("results/" + filename + "*-timingH.csv"):
        timingH = pd.read_csv(file)
        timingH = timingH.groupby("Options").agg([np.mean, confidenceInterval])
        ax.errorbar(timingH.index.values, timingH["Time"]["mean"],
                    yerr=timingH["Time"]["confidenceInterval"],
                    label="Time elapsed (%s)" % h_label)
    for ax in axs:
        ax.legend(loc="best")
        ax.set_xlabel(xlabel)

    fig.savefig("results/%s.png" % filename)

def confidenceInterval(x):
    std = x.std()
    count = x.count()
    return 1.96*std/math.sqrt(count)

def fairness():
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq
    Random.seed(2)
    options = UniformRandomVariable(defaultOptions, defaultOptions)
    generate_input_datas(K=10)
    generator = GeneratorForModel(servers, serviceProviders,
                                  options, containers, [cpu, ram], bandwidth, [cpuReq, ramReq])
    generator.generate()  # TODO make multithread by not using a singleton (can I?)
    npp = NetworkProvider().getInstance()
    npp.makePlacement(1)

    df = pd.DataFrame(columns=["ServiceProvider", "CPU", "RAM", "Utility", "Width"])
    for i, sp in enumerate(npp.getServiceProviders(), 1):
        for j, opt in enumerate(sp.getOptions(), 1):
            cpu = opt.getCpuReq()
            ram = opt.getRamReq()
            utility = opt.getBandwidthSaving()
            df.loc[len(df)] = {"ServiceProvider": i, "CPU": cpu, "RAM": ram, "Utility": utility, "Width":
                               70 if opt is sp.getDefaultOption() else 10
                               }
    df.to_csv("results/fairness.csv")

def fairness_graph():
    if len(glob.glob("results/fairness.csv")) == 0:
        fairness()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
    monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':']) * cycler('marker',['^', ',', '.']))
    #for ax in axs:
    #    ax.set_prop_cycle(monochrome)

    df = pd.read_csv("results/fairness.csv")
    ax = axs[0]
    for i in df["ServiceProvider"].unique():
        ax.scatter(df.loc[df["ServiceProvider"] == i]["ServiceProvider"],
                   df.loc[df["ServiceProvider"] == i]["Utility"],
                   df.loc[df["ServiceProvider"] == i]["Width"])
    ax.set_xlabel("Service provider ID")
    ax.set_ylabel("Utility")
    ax = axs[1]
    for i in df["ServiceProvider"].unique():
        ax.scatter(df.loc[df["ServiceProvider"] == i]["CPU"],
                   df.loc[df["ServiceProvider"] == i]["RAM"],
                   df.loc[df["ServiceProvider"] == i]["Width"])
    ax.set_xlabel("CPUs")
    ax.set_ylabel("RAM (Mb)")
    fig.savefig("results/fairness.png")

def fairness_graph2():
    global maxxCpu, maxxRam
    print(maxxCpu, maxxRam)
    if len(glob.glob("results/fairness.csv")) == 0:
        fairness()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
    monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':']) * cycler('marker',['^', ',', '.']))
    #for ax in axs:
    #    ax.set_prop_cycle(monochrome)

    df = pd.read_csv("results/fairness.csv")
    ax = axs[0]
    for i in df["ServiceProvider"].unique():
        ax.scatter(df.loc[(df["ServiceProvider"] == i) & (df["Width"] == 10)]["ServiceProvider"],
                   df.loc[(df["ServiceProvider"] == i) & (df["Width"] == 10)]["Utility"],
                   c="k")
    ax.scatter(df.loc[df["Width"] == 70]["ServiceProvider"],
           df.loc[df["Width"] == 70]["Utility"],
           c="k", marker="x", s=70)
    ax.set_xlabel("Service provider ID")
    ax.set_ylabel("Utility")
    ax = axs[1]
    ax.scatter(df.loc[df["Width"] == 70]["CPU"],
               df.loc[df["Width"] == 70]["RAM"], c="k")
    ax.set_ylim([0, maxxRam])
    ax.set_xlim([0, maxxCpu])
    ax.set_xlabel("CPUs")
    ax.set_ylabel("RAM (Mb)")
    fig.subplots_adjust(wspace=0.35)
    fig.savefig("results/fairness.png")

if __name__ == "__main__":
    os.system("rm -rf tresults/*")
    #simple(200)
    #grouped(20, 10)
    #simpleHeuristic(10)
    #groupedTogether(20, 10)

    #groupedTogetherSaveDifferentServersFixedCC(20, 10, [1, 2, 4, 8])
    #groupedTogetherSaveFixedServersDiferentCC(20, 10, [1, 2, 4, 8])
    #make_graph_from_file("cmpILPH-Options-VaryingContainers", "containers")
    #make_graph_from_file("cmpILPH-Options-VaryingServers", "servers")
    #groupedTogetherSaveVarOptionsFixedServersFixedCC(20, 10)
    groupedTogetherSaveFixedOptionsVarServersFixedCC(20, 16)
    #groupedTogetherSaveFixedOptionsFixedServersVarCC(20, 16)
    make_graph_from_file("cmpILPH-Options", "Options", "Number of options per service providers")
    make_graph_from_file("cmpILPH-Servers", "Servers", "Number of servers")
    make_graph_from_file("cmpILPH-Containers", "Containers", "Number of containers per option")

    #make_graph_from_file("cmpHN-Options", "Options", "Number of options per service providers", ilp_label="Naive")
    #generate_input_datas()
    #fairness_graph2()
