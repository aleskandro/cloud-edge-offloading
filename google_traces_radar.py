import pandas as pd
import time

from Generator.GeneratorForModelGoogle import *
from Random.NormalRandomVariable import *
from Random.UniformRandomVariable import *
from Random.ResourceDependentRandomVariable import *
import utils.radar_chart as radar_chart
import glob

maxxCpu = 30
maxxRam = 30000
defaultOptions = 5
Random.seed(5)

def generate_input_datas(avgCpu=1, avgRam=1, avgServers=1, avgContainers=8, avgServiceProviders=50, K=1.8):
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
    df = pd.read_csv(filename, index_col=0)
    print(df)
    radar_chart.make_radar_chart(df)

#os.system("rm -rf results/*")
#simple(200)
#grouped(20, 10)
#simpleHeuristic(25)
#groupedTogether(20, 10)
#radar_chart()
if __name__ == "__main__":
    Random.seed(6)
    if not len(glob.glob("results/radar_plot.csv")) > 0:
        make_datas_var_options_var_sps()

    make_radar_chart()

    Random.seed(6)
    if not len(glob.glob("results/radar_plot_relative.csv")) > 0:
        make_datas_var_options_var_sps(filename="results/radar_plot_relative.csv", ret_func=NetworkProvider()
                                       .getInstance().getRelativeBandwidthSaving)

    make_radar_chart("results/radar_plot_relative.csv")