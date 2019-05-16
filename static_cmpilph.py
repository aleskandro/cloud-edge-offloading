import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
import os
import time
import glob
from cycler import cycler
import matplotlib.pyplot as plt

from Generator.GeneratorBwConcave import *
from Generator.GeneratorForModel import *

from Random.NormalRandomVariable import *
from Random.UniformRandomVariable import *
from Random.ResourceDependentRandomVariable import *

maxxCpu = 30
maxxRam = 30000


def confidence_interval(x):
    return 1.96 * x.std() / math.sqrt(x.count())


def generate_input_datas(avgCpu=32, avgRam=32768, avgServers=8, avgContainers=8, avgServiceProviders=50, avgOptions=5,
                         K=2):
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq, maxxCpu, maxxRam, options

    maxxCpu = avgCpu * avgServers
    maxxRam = avgRam * avgServers

    servers = UniformRandomVariable(avgServers, avgServers)
    ram = NormalRandomVariable(avgRam, 0)
    cpu = NormalRandomVariable(avgCpu, 0)

    serviceProviders = UniformRandomVariable(avgServiceProviders, avgServiceProviders)
    bandwidth = ResourceDependentRandomVariable(UniformRandomVariable(1, 5))
    containers = UniformRandomVariable(avgContainers, avgContainers)

    ramReq = UniformRandomVariable(0, K * (avgRam * avgServers) / (avgContainers * avgServiceProviders), False)
    cpuReq = UniformRandomVariable(0, K * (avgCpu * avgServers) / (avgContainers * avgServiceProviders), False)

    options = UniformRandomVariable(avgOptions, avgOptions)
    return servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq, options


def simple_heuristic(max_it, varying_func, placement_func=NetworkProvider().getInstance().makePlacement):
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq
    bw_opts = pd.DataFrame(columns=["BandwidthSaving", "Options"])
    rr_opts = pd.DataFrame(columns=["Options", "CPU", "RAM"])
    timing = pd.DataFrame(columns=["Options", "Time"])
    i = 1
    while i <= max_it + 1:
        varying_func(i)
        generator = GeneratorForModel(servers, serviceProviders,
                                      options, containers, [cpu, ram], bandwidth, [cpuReq, ramReq])
        generator.generate()  # TODO make multithread by not using a singleton (can I?)
        npp = NetworkProvider().getInstance()
        t1 = time.time()
        placement_func(i)
        t2 = time.time()
        timing.loc[len(timing)] = {"Options": i, "Time": t2 - t1}
        bw_opts.loc[len(bw_opts)] = {"Options": i, "BandwidthSaving": npp.getBandwidthSaving()}
        rr_opts.loc[len(rr_opts)] = {"Options": i, "CPU": npp.getRemainingResources()[0],
                                     "RAM": npp.getRemainingResources()[1]}
        i *= 2
    return bw_opts, rr_opts, timing


def grouped_heuristic(runs, max_it, varying_func, func_placement=NetworkProvider().getInstance().makePlacement):
    bw_optss = pd.DataFrame(columns=["Options", "BandwidthSaving"])
    rr_optss = pd.DataFrame(columns=["Options", "CPU", "RAM"])
    timingg = pd.DataFrame(columns=["Options", "Time"])
    for i in range(0, runs):
        Random.seed(i)
        bw_opts, rr_opts, timing = simple_heuristic(max_it, varying_func, func_placement)
        bw_optss = bw_optss.append(bw_opts)
        rr_optss = rr_optss.append(rr_opts)
        timingg = timingg.append(timing)
    return bw_optss, rr_optss, timingg


def simple_ilp(max_it, varying_func):
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq, options
    timing = pd.DataFrame(columns=["Options", "Time"])
    i = 1
    while i <= max_it + 1:
        varying_func(i)
        generator = GeneratorBwConcave(servers, serviceProviders, options, containers, [cpu, ram], bandwidth,
                                       [cpuReq, ramReq])
        generator.generate()
        t1 = time.time()
        os.system("glpsol --math modelglpk.mod -d scenario.dat --tmlim 600")
        t2 = time.time()
        timing.loc[len(timing)] = {"Options": i, "Time": t2 - t1}
        i *= 2
    return timing


def grouped_ilp(runs, max_it, varying_func):
    timingg = pd.DataFrame(columns=["Options", "Time"])
    for i in range(0, runs):
        Random.seed(i)
        timing = simple_ilp(max_it, varying_func)
        timingg = timingg.append(timing)
    bw_opts = pd.read_csv("tresults/bandwidthByAvgOptions.csv", header=None,
                          names=["Servers", "Containers", "Options", "BandwidthSaving"])
    rr_opts = pd.read_csv("tresults/remainingResourcesByAvgOptions.csv", header=None,
                          names=["Servers", "Containers", "Options", "CPU", "RAM"])
    return bw_opts, rr_opts, timingg


def execute_simulations(runs, maxOpts, varying_func):
    filename = "cmpILPH-Containers"
    os.system("rm -rf tresults/*")
    if len(glob.glob("results/%s*ILP.csv" % filename)) == 0:
        bwOptsILP, rrOptsILP, timingILP = grouped_ilp(runs, maxOpts, varying_func)
        save_to_file(filename, "ILP", bwOptsILP, rrOptsILP, timingILP)
    if len(glob.glob("results/%s*H.csv" % filename)) == 0:
        bwOptsH, rrOptsH, timingH = grouped_heuristic(runs, maxOpts, varying_func)
        save_to_file(filename, "H", bwOptsH, rrOptsH, timingH)

    if len(glob.glob("results/%s*N.csv" % filename)) == 0:
        bwOptsN, rrOptsN, timingN = grouped_heuristic(runs, maxOpts, varying_func,
                                                      NetworkProvider().getInstance().make_placement_naive)
        save_to_file(filename, "N", bwOptsN, rrOptsN, timingN)


def save_to_file(filename, suffix, bwOpts, rrOpts, timing):
    bwOpts.to_csv("results/%s-bwOpts%s.csv" % (filename, suffix))
    rrOpts.to_csv("results/%s-rrOpts%s.csv" % (filename, suffix))
    timing.to_csv("results/%s-timing%s.csv" % (filename, suffix))


def make_graph_from_file(filename, ilp_key, xlabel, ilp_label="ILP", h_label="MOEPH"):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10,10))
    #monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':']) * cycler('marker',['^', ',', '.']))
    #for ax in axs:
    #    ax.set_prop_cycle(monochrome)

    ax = axs[0]
    max_y = 0
    ax.set_ylim([0, max_y])
    ax.set_ylabel("Utility")
    for file in glob.glob("results/" + filename + "*-bwOptsILP.csv"):
        bwOptsILP = pd.read_csv(file)
        bwOptsILP["BandwidthSaving"] = bwOptsILP["BandwidthSaving"].astype(float)
        bwOptsILP = bwOptsILP.groupby(ilp_key).agg([np.mean, confidence_interval])
        ax.errorbar(bwOptsILP.index.values, bwOptsILP["BandwidthSaving"]["mean"],
                    yerr=bwOptsILP["BandwidthSaving"]["confidenceInterval"],
                    label="Utility (%s)" % ilp_label)
        max_y = max(math.ceil(bwOptsILP["BandwidthSaving"]["mean"].max()) + 2, max_y)
        ax.set_ylim([0, max_y])
    for file in glob.glob("results/" + filename + "*-bwOptsH.csv"):
        bwOptsH = pd.read_csv(file)
        bwOptsH["BandwidthSaving"] = bwOptsH["BandwidthSaving"].astype(float)
        bwOptsH = bwOptsH.groupby("Options").agg([np.mean, confidence_interval])
        ax.errorbar(bwOptsH.index.values, bwOptsH["BandwidthSaving"]["mean"],
                    yerr=bwOptsH["BandwidthSaving"]["confidenceInterval"],
                    label="Utility (%s)" % h_label)
        max_y = max(math.ceil(bwOptsH["BandwidthSaving"]["mean"].max()) + 2, max_y)
        ax.set_ylim([0, max_y])
    ax = axs[1]
    ax.set_ylim([0, 30])
    ax.set_ylabel("Available resources after placement (%)")
    width = 0.5
    for file in glob.glob("results/" + filename + "*-rrOptsILP.csv"):
        rrOptsILP = pd.read_csv(file)
        rrOptsILP = rrOptsILP.groupby(ilp_key).agg([np.mean, confidence_interval])
        ax.bar(rrOptsILP.index.values - 3*width*np.array(rrOptsILP.index.values)/8, rrOptsILP["CPU"]["mean"]*100, width*np.array(rrOptsILP.index.values)/4,
                    yerr=rrOptsILP["CPU"]["confidenceInterval"]*100,
                    label="CPU (%s)" % ilp_label, align="edge")
        ax.bar(rrOptsILP.index.values - 1*width*np.array(rrOptsILP.index.values)/8, rrOptsILP["RAM"]["mean"]*100, width*np.array(rrOptsILP.index.values)/4,
                    yerr=rrOptsILP["RAM"]["confidenceInterval"]*100,
                    label="RAM (%s)" % ilp_label, align="edge")

    for file in glob.glob("results/" + filename + "*-rrOptsH.csv"):
        rrOptsH = pd.read_csv(file)
        rrOptsH = rrOptsH.groupby("Options").agg([np.mean, confidence_interval])
        ax.bar(rrOptsH.index.values + 1*width*np.array(rrOptsH.index.values)/8, rrOptsH["CPU"]["mean"]*100, width*np.array(rrOptsH.index.values)/4,
                    yerr=rrOptsH["CPU"]["confidenceInterval"]*100,
                    label="CPU (%s)" % h_label, align="edge")
        ax.bar(rrOptsH.index.values + 3*width*np.array(rrOptsH.index.values)/8, rrOptsH["RAM"]["mean"]*100, width*np.array(rrOptsH.index.values)/4,
                    yerr=rrOptsH["RAM"]["confidenceInterval"]*100,
                    label="RAM (%s)" % h_label, align="edge")

    ax = axs[2]
    ax.set_ylabel("Time elapsed (s)")
    for file in glob.glob("results/" + filename + "*-timingILP.csv"):
        timingILP = pd.read_csv(file)
        timingILP = timingILP.groupby("Options").agg([np.mean, confidence_interval])
        ax.errorbar(timingILP.index.values, timingILP["Time"]["mean"],
                    yerr=timingILP["Time"]["confidenceInterval"],
                    label="Time elapsed (%s)" % ilp_label)

    for file in glob.glob("results/" + filename + "*-timingH.csv"):
        timingH = pd.read_csv(file)
        timingH = timingH.groupby("Options").agg([np.mean, confidence_interval])
        ax.errorbar(timingH.index.values, timingH["Time"]["mean"],
                    yerr=timingH["Time"]["confidenceInterval"],
                    label="Time elapsed (%s)" % h_label)
    for ax in axs:
        ax.legend(loc="best")
        ax.set_xlabel(xlabel)
        ax.set_xscale('log', basex=2)
        formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
        ax.xaxis.set_major_formatter(formatter)

    fig.savefig("results/%s.png" % filename)


if __name__ == "__main__":
    os.system("rm -rf tresults/*")
    execute_simulations(20, 10, lambda x: generate_input_datas(avgOptions=x))
    execute_simulations(20, 16, lambda x: generate_input_datas(avgServers=x))
    execute_simulations(20, 16, lambda x: generate_input_datas(avgContainers=x))
    make_graph_from_file("cmpILPH-Options", "Options", "Number of options per service providers")
    make_graph_from_file("cmpILPH-Servers", "Servers", "Number of servers")
    make_graph_from_file("cmpILPH-Containers", "Containers", "Number of containers per option")
