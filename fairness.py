import static_cmpilph as cmpilph
import numpy.random as Random
from Model.NetworkProvider import NetworkProvider
from Generator.GeneratorForModel import GeneratorForModel
import pandas as pd
import glob
import matplotlib.pyplot as plt
import cycler
plt.rcParams.update({'font.size': 12})
def fairness():
    Random.seed(2)
    cmpilph.generate_input_datas(K=10)
    generator = GeneratorForModel(cmpilph.servers, cmpilph.serviceProviders,
                                  cmpilph.options, cmpilph.containers, [cmpilph.cpu, cmpilph.ram], cmpilph.bandwidth, [cmpilph.cpuReq, cmpilph.ramReq])
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
    cmpilph.generate_input_datas()
    if len(glob.glob("results/fairness.csv")) == 0:
        fairness()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
    #monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':']) * cycler('marker',['^', ',', '.']))
    #for ax in axs:
    #    ax.set_prop_cycle(monochrome)

    df = pd.read_csv("results/fairness.csv")
    ax = axs[0]
    first = True
    for i in df["ServiceProvider"].unique():
        ax.scatter(df.loc[(df["ServiceProvider"] == i) & (df["Width"] == 10)]["Utility"],
                   df.loc[(df["ServiceProvider"] == i) & (df["Width"] == 10)]["ServiceProvider"],
                   c="#777777", s=20, label="Available option" if first else None)
        first = False
    ax.scatter(df.loc[df["Width"] == 70]["Utility"],df.loc[df["Width"] == 70]["ServiceProvider"],
                c="k", marker="x", s=70, label="Chosen option")
    ax.set_xlabel("Utility")
    ax.set_ylabel("Service provider ID")
    ax.legend(loc="best")
    ax = axs[1]
    ax.scatter(df.loc[df["Width"] == 70]["CPU"],
               df.loc[df["Width"] == 70]["RAM"], c="#777777", s=20, label="Chosen option")
    ax.set_ylim([0, cmpilph.maxxRam])
    ax.set_xlim([0, cmpilph.maxxCpu])
    ax.set_xlabel("CPUs")
    ax.set_ylabel("RAM (Mb)")
    ax.legend(loc="best")
    fig.subplots_adjust(wspace=0.35)
    fig.savefig("results/fairness.eps", dpi=300)


if __name__ == "__main__":
    fairness_graph()

