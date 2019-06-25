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

def usage_vectors(simulate, naive=False, get_best_host=NetworkProvider().getInstance().getBestHost, filename="usage_vectors.csv"):
    Random.seed(2)
    cmpilph.generate_input_datas(K=2, avgCpu=1, avgRam=1)
    generator = GeneratorForModelGoogle(cmpilph.servers, cmpilph.serviceProviders,
                                  cmpilph.options, cmpilph.containers, [cmpilph.cpu, cmpilph.ram], cmpilph.bandwidth, [cmpilph.cpuReq, cmpilph.ramReq])
    generator.generate()  # TODO make multithread by not using a singleton (can I?)
    npp = NetworkProvider().getInstance()
    if not simulate:
        return
    if naive:
        npp.make_placement_naive()
    else:
        npp.makePlacement(1, get_best_host=get_best_host)

    df = pd.DataFrame(columns=["Node", "CPU", "RAM"])
    for i, sp in enumerate(npp.getServiceProviders()):
        if sp.getDefaultOption() is None:
            continue

        for container in sp.getDefaultOption().getContainers():
            cpu = container.getCpuReq()
            ram = container.getRamReq()
            node = npp.getServers().index(container.getServer())
            df.loc[len(df)] = {"Node": node, "CPU": cpu, "RAM": ram}
    if naive:
        df.to_csv("results/usage_vectors_naive.csv")
    else:
        df.to_csv("results/" + filename)



def usage_vectors_graph():
    cmpilph.generate_input_datas()
    simulate = len(glob.glob("results/usage_vectors.csv")) == 0
    usage_vectors(simulate, get_best_host=NetworkProvider().getInstance().getBestHost, filename="usage_vectors.csv")
    simulate = len(glob.glob("results/usage_vectors_naive.csv")) == 0
    usage_vectors(simulate, True)
    simulate = len(glob.glob("results/usage_vectors_min.csv")) == 0
    usage_vectors(simulate, get_best_host=NetworkProvider().getInstance().getBestHostMin, filename="usage_vectors_min.csv")
    simulate = len(glob.glob("results/usage_vectors_scalar.csv")) == 0
    usage_vectors(simulate, get_best_host=NetworkProvider().getInstance().getBestHostScalarProduct, filename="usage_vectors_scalar.csv")

    #monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':']) * cycler('marker',['^', ',', '.']))
    #for ax in axs:
    #    ax.set_prop_cycle(monochrome)

    df = pd.read_csv("results/usage_vectors.csv")
    #df_naive = pd.read_csv("results/usage_vectors_naive.csv")
    fig, axs = plt.subplots(nrows=int(df["Node"].max()) + 1, ncols=1, figsize=(10,60))
    def quiv(df, color="r", label="EdgeMORE"):
        last_coordinate = [(0,0)] * (int(df["Node"].max()) + 1)
        first = [True] * (int(df["Node"].max()) + 1)
        for index, row in df.iterrows():
            ax = axs[int(row["Node"])]
            ax.quiver([last_coordinate[int(row["Node"])][0]],
                      [last_coordinate[int(row["Node"])][1]],
                      [row["CPU"]],
                      [row["RAM"]], angles='xy', scale_units='xy', scale=1, width=0.0025,
                      color=[color], label=(label if first[int(row["Node"])] else None))
            first[int(row["Node"])] = False
            last_coordinate[int(row["Node"])] = (last_coordinate[int(row["Node"])][0]
                                                 + row["CPU"], last_coordinate[int(row["Node"])][1] + row["RAM"])
    quiv(df, "r", "EdgeMORE (max)")
    quiv(pd.read_csv("results/usage_vectors_min.csv"), "g", "EdgeMORE (min)")
    quiv(pd.read_csv("results/usage_vectors_scalar.csv"), "y", "EdgeMORE (scalar)")
    quiv(pd.read_csv("results/usage_vectors_naive.csv"), "b", "Naive")
    #axs[0].quiver([0], [0], [1], [1], angles='xy', scale_units='xy', scale=1)
    #axs[0].quiver([1], [1], [2], [3], angles='xy', scale_units='xy', scale=1)
    #axs[0].quiver([3], [4], [4], [9], angles='xy', scale_units='xy', scale=1)
    #axs[0].set_xlim(0, 10)
    #axs[0].set_ylim(0, 10)
    for i,ax in enumerate(axs):
        ax.set_ylim(0,NetworkProvider().getInstance().getServers()[i].getTotalRam())
        ax.set_xlim(0,NetworkProvider().getInstance().getServers()[i].getTotalCpu())
        ax.set_xlabel("CPU")
        ax.set_ylabel("RAM")
        ax.legend()
    fig.savefig("results/usage_vectors.pdf", dpi=300)


if __name__ == "__main__":
    usage_vectors_graph()

