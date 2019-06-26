import static_cmpilph as cmpilph
import numpy.random as Random
from Model.NetworkProvider import NetworkProvider
from Generator.GeneratorForModel import GeneratorForModel
from Generator.GeneratorForModelGoogle import GeneratorForModelGoogle
import pandas as pd
import glob
import matplotlib.pyplot as plt
import cycler
plt.rcParams.update({'font.size': 12.5, 'font.family': 'serif'})


def iterations_report(simulate, get_best_host=NetworkProvider().getInstance().getBestHost, filename="iterations_report.csv"):
    Random.seed(2)
    cmpilph.generate_input_datas(K=2, avgCpu=1, avgRam=1)
    generator = GeneratorForModelGoogle(cmpilph.servers, cmpilph.serviceProviders,
                                  cmpilph.options, cmpilph.containers, [cmpilph.cpu, cmpilph.ram], cmpilph.bandwidth, [cmpilph.cpuReq, cmpilph.ramReq])
    generator.generate()  # TODO make multithread by not using a singleton (can I?)
    npp = NetworkProvider().getInstance()
    if not simulate:
        return
    df = npp.makePlacement(1, get_best_host=get_best_host, collect_iterations_report=True)

    df.to_csv("results/" + filename)


def iterations_report_graph():
    cmpilph.generate_input_datas()
    simulate = len(glob.glob("results/iterations_report.csv")) == 0
    iterations_report(simulate)

    #monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':']) * cycler('marker',['^', ',', '.']))
    #for ax in axs:
    #    ax.set_prop_cycle(monochrome)

    df = pd.read_csv("results/iterations_report.csv")
    #df_naive = pd.read_csv("results/usage_vectors_naive.csv")
    fig, axs = plt.subplots(nrows=(len(df.columns) - 4)/2 + 2, ncols=1, figsize=(10, 60))

    # Utility and expected utility
    ax = axs[0]
    ax.errorbar(df["Iteration"], df["Utility"],
                #yerr=100*bwOpts["BandwidthSaving"]["confidence_interval"]/aavgServiceProviders,
                label="Utility")
    ax.errorbar(df["Iteration"], df["ExpectedUtility"],
                #yerr=100*bwOpts["BandwidthSaving"]["confidence_interval"]/aavgServiceProviders,
                label="Expected utility")
    ax.set_ylabel("Utility")

    ax = axs[1]
    ax.errorbar(df["Iteration"], df["BestJumpEfficiency"], label="Best jump efficiency")
    ax.set_ylabel("Efficiency")

    npp = NetworkProvider().getInstance()
    for index, server in enumerate(npp.getServers()):
        ax = axs[2 + index]
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage of occupied resources")
        ax.errorbar(df["Iteration"], df["%d_CPU" % index] * 100, label="CPU")
        ax.errorbar(df["Iteration"], df["%d_RAM" % index] * 100, label="RAM")

    for i, ax in enumerate(axs):
        ax.set_xlabel("Iteration")
        ax.legend()
    fig.savefig("results/iterations_report.pdf", dpi=300)


if __name__ == "__main__":
    iterations_report_graph()

