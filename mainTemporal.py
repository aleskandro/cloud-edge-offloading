import numpy.random as Random
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import glob

from Generator.Generator import *
from Generator.GeneratorBwConcave import *
from Generator.GeneratorForModel import *

from Random.NormalRandomVariable import *
from Random.UniformRandomVariable import *
from Random.PoissonRandomVariable import *
from Random.ResourceDependentRandomVariable import *
from Random.ExponentialRandomVariable import *

def generate_input_datas(maxCpu=30, maxRam=30000, avgServers=8, avgContainers=8, avgServiceProviders=5, K=1.6, trate = 1, ttime_window=20, execution_time_scale=2):
    Random.seed(1)
    avgRam = maxRam/avgServers
    avgCpu = maxCpu/avgServers
    # avgBandwidth = 100
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq, rate, time_window, time_limit
    global serviceProviders, execution_time

    servers = UniformRandomVariable(avgServers, avgServers)
    ram = NormalRandomVariable(avgRam, 0)
    cpu = NormalRandomVariable(avgCpu, 0)

    serviceProviders = UniformRandomVariable(avgServiceProviders, avgServiceProviders)
    # bandwidth = NormalRandomVariable(avgBandwidth, 50)
    bandwidth = ResourceDependentRandomVariable(UniformRandomVariable(1,5))
    containers = UniformRandomVariable(avgContainers, avgContainers)

    ramReq = UniformRandomVariable(0, K * (avgRam * avgServers) / (avgContainers * avgServiceProviders))
    cpuReq = UniformRandomVariable(0, K * (avgCpu * avgServers) / (avgContainers * avgServiceProviders))


    rate = trate  # req/slot
    time_window = ttime_window  # min/slot
    time_limit = 200  # limit time for batch execution

    serviceProviders = PoissonRandomVariable(time_window * rate)
    execution_time = ExponentialRandomVariable(execution_time_scale)

    return servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq

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

def varying_rate(runs, maxOpts, rates):
    for rate in rates:
        if len(glob.glob("results/batched-rate-%d-*" % rate)) > 0:
            continue
        generate_input_datas(trate=rate)
        bwOpts, rrOpts, timing, activeServices = groupedHeuristic(runs, maxOpts, False)

        bwOpts = bwOpts.groupby("t").agg([np.mean, confidenceInterval])
        rrOpts = rrOpts.groupby("t").agg([np.mean, confidenceInterval])
        timing = timing.groupby("t").agg([np.mean, confidenceInterval])
        activeServices["Services"] = activeServices["Services"].astype(float)
        activeServices = activeServices.groupby("t").agg([np.mean, confidenceInterval])

        bwOpts.to_csv("results/batched-rate-%d-bwopts.csv" % rate)
        rrOpts.to_csv("results/batched-rate-%d-rropts.csv" % rate)
        timing.to_csv("results/batched-rate-%d-timing.csv" % rate)
        activeServices.to_csv("results/batched-rate-%d-services.csv" % rate)
    make_graph_from_file("batched-rate-*")

def varying_ex_time_scale(runs, maxOpts, ex_time_scales):
    for execution_time_scale in ex_time_scales:
        if len(glob.glob("results/batched-scale-%d-*")) > 0:
            continue
        generate_input_datas(execution_time_scale=execution_time_scale)

        bwOpts, rrOpts, timing, activeServices = groupedHeuristic(runs, maxOpts, False)

        bwOpts = bwOpts.groupby("t").agg([np.mean, confidenceInterval])
        rrOpts = rrOpts.groupby("t").agg([np.mean, confidenceInterval])
        timing = timing.groupby("t").agg([np.mean, confidenceInterval])
        activeServices["Services"] = activeServices["Services"].astype(float)
        activeServices = activeServices.groupby("t").agg([np.mean, confidenceInterval])

        bwOpts.to_csv("results/batched-scale-%d-bwopts.csv" % execution_time_scale)
        rrOpts.to_csv("results/batched-scale-%d-rropts.csv" % execution_time_scale)
        timing.to_csv("results/batched-scale-%d-timing.csv" % execution_time_scale)
        activeServices.to_csv("results/batched-scale-%d-services.csv" % execution_time_scale)
    make_graph_from_file("batched-scale-*")
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

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 15))
    ax = axs[0]
    ax.set_ylim([0, math.ceil(bwOpts["BandwidthSaving"]["mean"].max())])
    ax.errorbar(bwOpts.index.values, bwOpts["BandwidthSaving"]["mean"], yerr=bwOpts["BandwidthSaving"]["confidenceInterval"], label="Bandwidth saving")
    ax.legend(loc="best")
    ax.set_xlabel("Time")
    ax = axs[1]
    ax.errorbar(rrOpts.index.values, rrOpts["CPU"]["mean"], yerr=rrOpts["CPU"]["confidenceInterval"], label="CPU")
    ax.errorbar(rrOpts.index.values, rrOpts["RAM"]["mean"], yerr=rrOpts["RAM"]["confidenceInterval"], label="RAM")
    ax.legend(loc="best")
    ax.set_ylim([0, 1])
    ax.set_xlabel("Time")
    ax = axs[2]
    ax.errorbar(timing.index.values, timing["Time"]["mean"], yerr=timing["Time"]["confidenceInterval"], label="Time elapsed")
    ax.legend(loc="best")
    ax.set_xlabel("Time")
    ax = axs[3]
    ax.errorbar(activeServices.index.values, activeServices["Services"]["mean"], yerr=activeServices["Services"]["confidenceInterval"], label="Active services")
    ax.legend(loc="best")
    ax.set_xlabel("Time")
    fig.savefig("results/output.png")

def make_graph_from_file(filename_regex):
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10,15))
    ylim = 0
    ax = axs[0]
    ax.set_ylabel("Utility")
    for file in glob.glob("results/" + filename_regex + "-bwopts.csv"):
        bwopts = pd.read_csv(file)
        varied_key = file.split("-")[2]
        varied_value = file.split("-")[3]
        ylim = max(ylim, math.ceil(bwopts["BandwidthSaving"]["mean"].max()))
        ax.set_ylim([0, ylim])
        ax.errorbar(bwopts.index.values, bwopts["BandwidthSaving"]["mean"], yerr=bwopts["BandwidthSaving"]["confidenceInterval"],
                    label="Bandwidth saving (%s = %s)" % (varied_key, varied_value))
    ax = axs[1]
    ax.set_ylabel("Available resources (%)")
    ax.set_ylim([0, 100])
    for file in glob.glob("results/" + filename_regex + "-rropts.csv"):
        rropts = pd.read_csv(file)
        varied_key = file.split("-")[2]
        varied_value = file.split("-")[3]
        ax.errorbar(rropts.index.values, rropts["CPU"]["mean"]*100, yerr=rropts["CPU"]["confidenceInterval"]*100,
                    label="CPUs (%s = %s)" % (varied_key, varied_value))
        ax.errorbar(rropts.index.values, rropts["RAM"]["mean"]*100, yerr=rropts["RAM"]["confidenceInterval"]*100,
                    label="RAM (%s = %s)" % (varied_key, varied_value))

    ax = axs[2]
    ax.set_ylabel("Time elapsed (s)")
    for file in glob.glob("results/" + filename_regex + "-timing.csv"):
        timing = pd.read_csv(file)
        varied_key = file.split("-")[2]
        varied_value = file.split("-")[3]
        ax.errorbar(timing.index.values, timing["Time"]["mean"], yerr=timing["Time"]["confidenceInterval"],
                    label="Time (s) (%s = %s)" % (varied_key, varied_value))

    ax = axs[3]
    ax.set_ylabel("Active services")
    for file in glob.glob("results/" + filename_regex + "-services.csv"):
        timing = pd.read_csv(file)
        varied_key = file.split("-")[2]
        varied_value = file.split("-")[3]
        ax.errorbar(timing.index.values, timing["Time"]["mean"], yerr=timing["Time"]["confidenceInterval"],
                    label="N. services (%s = %s)" % (varied_key, varied_value))

    for ax in axs:
        ax.set_xlabel("Time (m)")
        ax.legend(loc="best")
    fig.savefig("results/"+ filename_regex[0:-2] + ".png")

generate_input_datas()
#groupedHeuristic(20, 10)
varying_rate(20, 10, [1, 2, 4, 8])
varying_ex_time_scale(20, 10, [2, 5, 30])
