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
plt.rcParams.update({'font.size': 12})

def confidence_interval(x):
    return 1.96 * x.std() / math.sqrt(x.count())


def generate_input_datas(avgCpu=32, avgRam=32768, avgServers=8, avgContainers=8, avgServiceProviders=50, avgOptions=5,
                         K=1.8):
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq, maxxCpu, maxxRam, options, Kk, aavgServiceProviders

    Kk = K

    maxxCpu = avgCpu * avgServers
    maxxRam = avgRam * avgServers
    aavgServiceProviders = avgServiceProviders

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


def simple_heuristic(varying_func, key, iterations, placement_func=NetworkProvider().getInstance().makePlacement):
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq
    bw_opts = pd.DataFrame(columns=["BandwidthSaving", key])
    rr_opts = pd.DataFrame(columns=[key, "CPU", "RAM"])
    timing = pd.DataFrame(columns=[key, "Time"])
    for i in iterations:
        varying_func(i)
        generator = GeneratorForModel(servers, serviceProviders,
                                      options, containers, [cpu, ram], bandwidth, [cpuReq, ramReq])
        generator.generate()  # TODO make multithread by not using a singleton (can I?)
        npp = NetworkProvider().getInstance()
        t1 = time.time()
        placement_func(i)
        t2 = time.time()
        timing.loc[len(timing)] = {key: i, "Time": t2 - t1}
        bw_opts.loc[len(bw_opts)] = {key: i, "BandwidthSaving": npp.getBandwidthSaving()}
        rr_opts.loc[len(rr_opts)] = {key: i, "CPU": npp.getRemainingResources()[0],
                                     "RAM": npp.getRemainingResources()[1]}
        i *= 2
    return bw_opts, rr_opts, timing


def grouped_heuristic(runs, varying_func, key, iterations, func_placement=NetworkProvider().getInstance().makePlacement):
    bw_optss = pd.DataFrame(columns=[key, "BandwidthSaving"])
    rr_optss = pd.DataFrame(columns=[key, "CPU", "RAM"])
    timingg = pd.DataFrame(columns=[key, "Time"])
    for i in range(0, runs):
        Random.seed(i)
        bw_opts, rr_opts, timing = simple_heuristic(varying_func, key, iterations, func_placement)
        bw_optss = bw_optss.append(bw_opts)
        rr_optss = rr_optss.append(rr_opts)
        timingg = timingg.append(timing)
    return bw_optss, rr_optss, timingg


def simple_ilp(varying_func, key, iterations):
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq, options, Kk
    timing = pd.DataFrame(columns=[key, "Time"])
    for i in iterations:
        varying_func(i)
        generator = GeneratorBwConcave(servers, serviceProviders, options, containers, [cpu, ram], bandwidth,
                                       [cpuReq, ramReq], K=Kk)
        generator.generate()
        t1 = time.time()
        os.system("glpsol --math modelglpk.mod -d scenario.dat --tmlim 600")
        t2 = time.time()
        timing.loc[len(timing)] = {key: i, "Time": t2 - t1}
        i *= 2
    return timing


def grouped_ilp(runs, varying_func, key, iterations):
    timingg = pd.DataFrame(columns=[key, "Time"])
    for i in range(0, runs):
        Random.seed(i)
        timing = simple_ilp(varying_func, key, iterations)
        timingg = timingg.append(timing)
    bw_opts = pd.read_csv("tresults/bandwidthByAvgOptions.csv", header=None,
                          names=["Servers", "Containers", "Options", "K", "BandwidthSaving"])
    rr_opts = pd.read_csv("tresults/remainingResourcesByAvgOptions.csv", header=None,
                          names=["Servers", "Containers", "Options", "K", "CPU", "RAM"])
    return bw_opts, rr_opts, timingg


def execute_simulations(runs, varying_func, key, iterations):
    filename = "cmpILPH-%s" % key
    os.system("rm -rf tresults/*")
    if len(glob.glob("results/%s*ILP.csv" % filename)) == 0:
        bwOptsILP, rrOptsILP, timingILP = grouped_ilp(runs, varying_func, key, iterations)
        save_to_file(filename, "ILP", bwOptsILP, rrOptsILP, timingILP)
    if len(glob.glob("results/%s*H.csv" % filename)) == 0:
        bwOptsH, rrOptsH, timingH = grouped_heuristic(runs, varying_func, key, iterations)
        save_to_file(filename, "H", bwOptsH, rrOptsH, timingH)

    if len(glob.glob("results/%s*N.csv" % filename)) == 0:
        bwOptsN, rrOptsN, timingN = grouped_heuristic(runs, varying_func, key, iterations,
                                                      NetworkProvider().getInstance().make_placement_naive)
        save_to_file(filename, "N", bwOptsN, rrOptsN, timingN)


def save_to_file(filename, suffix, bwOpts, rrOpts, timing):
    bwOpts.to_csv("results/%s-bwOpts%s.csv" % (filename, suffix))
    rrOpts.to_csv("results/%s-rrOpts%s.csv" % (filename, suffix))
    timing.to_csv("results/%s-timing%s.csv" % (filename, suffix))


def make_graph_from_file(filename, group_key, xlabel, log=True, width=0.5):
    global aavgServiceProviders
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10,10))
    #monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':']) * cycler('marker',['^', ',', '.']))
    #for ax in axs:
    #    ax.set_prop_cycle(monochrome)

    ax = axs[0]
    max_y = 0
    ax.set_ylim([0, max_y])
    ax.set_ylabel("Utility (%)")
    for label, regex in [("Utility (Optimal)", "*-bwOptsILP.csv"), ("Utility (EdgeMORE)", "*-bwOptsH.csv"), ("Utility (Naive)", "*-bwOptsN.csv")]:
        for file in glob.glob("results/" + filename + regex):
            print(label, regex)
            bwOpts = pd.read_csv(file)
            bwOpts["BandwidthSaving"] = bwOpts["BandwidthSaving"].astype(float)
            bwOpts = bwOpts.groupby(group_key).agg([np.mean, confidence_interval])
            ax.errorbar(bwOpts.index.values, 100*bwOpts["BandwidthSaving"]["mean"]/aavgServiceProviders,
                        yerr=100*bwOpts["BandwidthSaving"]["confidence_interval"]/aavgServiceProviders,
                        label=label)
            #max_y = max(math.ceil(bwOpts["BandwidthSaving"]["mean"].max()/aavgServiceProviders) + 2, max_y)
            #ax.set_ylim([0, max_y])
    ax.set_ylim([0, 50])
    ax = axs[1]
    max_y = 30
    ax.set_ylabel("Available resources after placement (%)")
    start = -5
    for label, regex in [("Optimal", "*-rrOptsILP.csv"), ("EdgeMORE", "*-rrOptsH.csv"), ("Naive", "*-rrOptsN.csv")]:
        for file in glob.glob("results/" + filename + regex):
            rrOpts = pd.read_csv(file)
            rrOpts = rrOpts.groupby(group_key).agg([np.mean, confidence_interval])
            max_y = max(max_y, rrOpts["CPU"]["mean"].max()*100 + 5)
            max_y = max(max_y, rrOpts["RAM"]["mean"].max()*100 + 5)
            if log:
                ax.bar(rrOpts.index.values + start*width*np.array(rrOpts.index.values)/12, rrOpts["CPU"]["mean"]*100, width*np.array(rrOpts.index.values)/6,
                            yerr=rrOpts["CPU"]["confidence_interval"]*100,
                            label="CPU (%s)" % label, align="edge")
                start += 2
                ax.bar(rrOpts.index.values + start*width*np.array(rrOpts.index.values)/12, rrOpts["RAM"]["mean"]*100, width*np.array(rrOpts.index.values)/6,
                            yerr=rrOpts["RAM"]["confidence_interval"]*100,
                            label="RAM (%s)" % label, align="edge")
                start += 2
            else:
                ax.bar(rrOpts.index.values + start*width/12, rrOpts["CPU"]["mean"]*100, width/6,
                            yerr=rrOpts["CPU"]["confidence_interval"]*100,
                            label="CPU (%s)" % label, align="edge")
                start += 2
                ax.bar(rrOpts.index.values + start*width/12, rrOpts["RAM"]["mean"]*100, width/6,
                            yerr=rrOpts["RAM"]["confidence_interval"]*100,
                            label="RAM (%s)" % label, align="edge")
                start += 2
    ax.set_ylim([0, max_y])


    ax = axs[2]
    ax.set_ylabel("Time elapsed (s)")
    for label, regex in [("Optimal", "*-timingILP.csv"), ("EdgeMORE", "*-timingH.csv"), ("Naive", "*-timingN.csv")]:
        for file in glob.glob("results/" + filename + regex):
            timingILP = pd.read_csv(file)
            timingILP = timingILP.groupby(group_key).agg([np.mean, confidence_interval])
            ax.errorbar(timingILP.index.values, timingILP["Time"]["mean"],
                        yerr=timingILP["Time"]["confidence_interval"],
                        label="Time elapsed (%s)" % label)

    for ax in axs:
        ax.legend(loc="best")
        ax.set_xlabel(xlabel)
        if log:
            ax.set_xscale('log', basex=2)
        formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
        ax.xaxis.set_major_formatter(formatter)

    fig.savefig("results/%s.eps" % filename, dpi=500)


if __name__ == "__main__":
    os.system("rm -rf tresults/*")
    generate_input_datas()
    execute_simulations(20, lambda x: generate_input_datas(avgOptions=x), "Options", [1, 2, 4, 8])
    execute_simulations(20, lambda x: generate_input_datas(avgServers=x), "Servers", [1, 2, 4, 8, 16])
    execute_simulations(20, lambda x: generate_input_datas(avgContainers=x), "Containers", [1, 2, 4, 8, 16])
    execute_simulations(20, lambda x: generate_input_datas(K=x), "K", [0.5, 1, 1.5, 1.8, 2])
    make_graph_from_file("cmpILPH-Options", "Options", "Number of options per service providers")
    make_graph_from_file("cmpILPH-Servers", "Servers", "Number of servers")
    make_graph_from_file("cmpILPH-Containers", "Containers", "Number of containers per option")
    make_graph_from_file("cmpILPH-K", "K", "K", log=False, width=0.15)
