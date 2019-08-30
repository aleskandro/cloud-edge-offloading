import static_cmpilph as cmpilph
import numpy.random as Random
from Model.NetworkProvider import NetworkProvider
from Generator.GeneratorForModel import GeneratorForModel
from Generator.GeneratorForModelGoogle import GeneratorForModelGoogle
from Generator.GeneratorForModelAlibaba import GeneratorForModelAlibaba
import pandas as pd
import glob
import matplotlib.pyplot as plt
import cycler
plt.rcParams.update({'font.size': 12.5, 'font.family': 'serif'})


def k_report(seed=2, sp_nb_max=50):
    Random.seed(seed)
    df = pd.DataFrame(columns=["n_sp", "k_cpu", "k_ram"])

    for sp_nb in range(1, sp_nb_max):
        cmpilph.generate_input_datas(K=2, avgCpu=800, avgRam=100, avgServiceProviders=sp_nb)
        generator = GeneratorForModelAlibaba(cmpilph.servers, cmpilph.serviceProviders,
                                      cmpilph.options, cmpilph.containers, [cmpilph.cpu, cmpilph.ram],
                                            cmpilph.bandwidth, [cmpilph.cpuReq, cmpilph.ramReq])
        generator.generate()  # TODO make multithread by not using a singleton (can I?)
        df = df.append({"n_sp": sp_nb, "k_cpu": generator.getK()[0], "k_ram": generator.getK()[1]})

    return df


def k_report_graph():
    simulate = len(glob.glob("results/googletracesK.csv")) == 0
    if simulate:
        dfs = []
        for i in range(1, 20):
            dfs.append(k_report(seed=i, sp_nb_max=50))
        df = pd.concat(dfs)
        df.to_csv("results/googletracesK.csv")
    #monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':']) * cycler('marker',['^', ',', '.']))
    #for ax in axs:
    #    ax.set_prop_cycle(monochrome)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    df = pd.read_csv("results/googletracesK.csv")

    ax = axs[0]
    ax.scatter(df["n_sp"], df.groupby("n_sp").mean()["k_cpu"], label="K_CPU")
    ax.set_ylabel("K_CPU")

    ax = axs[1]
    ax.scatter(df["n_sp"], df.groupby("n_sp").mean()["k_ram"], label="K_RAM")
    ax.set_ylabel("K_RAM")

    for i, ax in enumerate(axs):
        ax.set_xlabel("Number of service providers")
        ax.legend()

    fig.savefig("results/googletracesK.pdf", dpi=300)


if __name__ == "__main__":
    k_report_graph()
