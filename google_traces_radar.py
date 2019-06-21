import pandas as pd
import time
import numpy as np

from Generator.GeneratorForModelGoogle import *
from Generator.GeneratorForModel import *
from Random.NormalRandomVariable import *
from Random.UniformRandomVariable import *
from Random.ResourceDependentRandomVariable import *
import utils.radar_chart as radar_chart
import glob
import os

maxxCpu = 30
maxxRam = 30000
defaultOptions = 5
Random.seed(5)
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

def generate_input_datas(avgCpu=1, avgRam=1, avgServers=8, avgContainers=8, avgServiceProviders=50, K=1.8):
    global maxxCpu, maxxRam

    maxxCpu = avgCpu * avgServers
    maxxRam = avgRam * avgServers
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

def simpleHeuristic(maxOpt, make_graph=True):
    bwOpts2 = pd.DataFrame(columns=["BandwidthSaving", "ServiceProviders"])
    rrOpts2 = pd.DataFrame(columns=["ServiceProviders", "CPU", "RAM"])
    timing = pd.DataFrame(columns=["ServiceProviders", "Time"])
    k = pd.DataFrame(columns=["ServiceProviders", "K"])
    options = UniformRandomVariable(10, 10)
    for i in range(1, maxOpt + 1):
        serviceProviders = UniformRandomVariable(i, i)
        generator = GeneratorForModelGoogle(servers, serviceProviders,
                                            options, containers, [cpu, ram], bandwidth, [cpuReq, ramReq], K=1)
        generator.generate()  # TODO make multithread by not using a singleton (can I?)
        npp = NetworkProvider().getInstance()
        print([str(item) for item in npp.getServiceProviders()])
        t1 = time.time()
        npp.makePlacement(i)
        t2 = time.time()
        timing.loc[len(timing)] = {"ServiceProviders": i, "Time": t2 - t1}
        bwOpts2.loc[len(bwOpts2)] = {"ServiceProviders": i, "BandwidthSaving": npp.getBandwidthSaving()}
        rrOpts2.loc[len(rrOpts2)] = {"ServiceProviders": i, "CPU": npp.getRemainingResources()[0], "RAM": npp.getRemainingResources()[1]}
        k.loc[len(k)] = {"ServiceProviders": i, "K": generator.getK()}
    return bwOpts2, rrOpts2, timing

def make_datas_var_options_var_sps_raw(maxSPs=160, maxOpts=8, max_runs=20, filename="results/radar_plot_raw.csv"):
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq
    generate_input_datas()
    npp = NetworkProvider().getInstance()
    df = pd.DataFrame(columns=["options", "service_providers", "utility", "K", "remaining_cpu", "remaining_ram", "time_elapsed"])
    for i in range(max_runs):
        Random.seed((i+1)*3)
        j = 10
        while j <= maxSPs:
            serviceProviders = UniformRandomVariable(j, j)
            k = 1

            while k <= maxOpts:
                options = UniformRandomVariable(k, k)
                generator = GeneratorForModelGoogle(servers, serviceProviders,
                                            options, containers, [cpu, ram], bandwidth, [cpuReq, ramReq], K=1)
                generator.generate()  # TODO make multithread by not using a singleton (can I?)
                #generator.save_to_csv(i)
                #npp.clean_cluster()
                t1 = time.time()
                npp.makePlacement(1)
                t2 = time.time()
                #generator.save_for_ilp()

                #os.system("glpsol --math modelglpk.mod -d scenario.dat --proxy 600")
                df.loc[len(df)] = {"options": k, "service_providers": j, "utility": npp.getBandwidthSaving(),
                                   "K": generator.getK(), "remaining_cpu": npp.getRemainingResources()[0],
                                   "remaining_ram": npp.getRemainingResources()[1], "time_elapsed": t2-t1}
                print(df)
                k *= 2
            j *= 2
    df.to_csv(filename)


def make_datas_var_options_var_sps(maxSPs=160, maxOpts=8, ret_func=NetworkProvider().getInstance().getBandwidthSaving, filename="results/radar_plot.csv"):
    global servers, ram, cpu, serviceProviders, bandwidth, containers, ramReq, cpuReq
    generate_input_datas()
    npp = NetworkProvider().getInstance()
    serviceProvidersNb = []
    i = 10
    while i <= maxSPs:
        serviceProvidersNb += [i]
        i *= 2

    optionsNb = []
    i = 1
    while i <= maxOpts:
        optionsNb += [i]
        i *= 2

    df = pd.DataFrame({
        'options': optionsNb,
    })

    def column_label(sps):
        return "%d SPs" % sps
    # add Columns to dataframe for each service provider
    for sp in serviceProvidersNb:
        df[column_label(sp)] = None

    for sps in serviceProvidersNb:
        serviceProviders = UniformRandomVariable(sps, sps)
        for opts in optionsNb:
            options = UniformRandomVariable(opts, opts)
            generator = GeneratorForModelGoogle(servers, serviceProviders,
                                            options, containers, [cpu, ram], bandwidth, [cpuReq, ramReq], K=1)
            generator.generate()  # TODO make multithread by not using a singleton (can I?)
            npp.makePlacement(1)
            df.loc[df["options"] == opts, column_label(sps)] = ret_func()
            print(df)

    df.to_csv(filename, index=False)

def make_radar_chart(filename="results/radar_plot.csv"):
    df = pd.read_csv(filename)
    print(df)
    radar_chart.make_radar_chart(df, filename.replace("csv", "eps"))

def make_csv_from_raw(filename="results/radar_plot_raw.csv", relative=True):
    df = pd.read_csv(filename)
    df = df.groupby(["options", "service_providers"]).agg(np.mean)
    def column_label(i):
        return "%d SPs" % i

    df_out = pd.DataFrame(columns=["options"] + [column_label(i) for i in list(df.index.levels[1])])
    for i in list(df.index.levels[0]): # options
        my_dict = {"options": i}
        for j in list(df.index.levels[1]): # service_providers
            my_dict[column_label(j)] = df.iloc[(df.index.get_level_values("options") == i)
                                 & (df.index.get_level_values("service_providers") == j)]["utility"].values[0]

            if relative:
                my_dict[column_label(j)] = my_dict[column_label(j)] / j

        print(my_dict)
        df_out.loc[len(df_out)] = my_dict

    df_out.to_csv("results/radar_plot_2.csv", index=False)

#os.system("rm -rf results/*")
#simple(200)
#grouped(20, 10)
#simpleHeuristic(25)
#groupedTogether(20, 10)
#radar_chart()
if __name__ == "__main__":
    # Radar plots with single run
    #Random.seed(6)
    #if not len(glob.glob("results/radar_plot.csv")) > 0:
    #    make_datas_var_options_var_sps()

    #make_radar_chart()

    #Random.seed(6)
    #if not len(glob.glob("results/radar_plot_relative.csv")) > 0:
    #    make_datas_var_options_var_sps(filename="results/radar_plot_relative.csv", ret_func=NetworkProvider()
    #                                   .getInstance().getRelativeBandwidthSaving)

    #make_radar_chart("results/radar_plot_relative.csv")

    os.system("rm -rf tresults/*")
    if not len(glob.glob("results/radar_plot_raw.csv")) > 0:
        make_datas_var_options_var_sps_raw()

    make_csv_from_raw()
    make_radar_chart("results/radar_plot_2.csv")
