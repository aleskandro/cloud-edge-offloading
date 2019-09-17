from Model.NetworkProvider import *
from Model.Container import *
from Model.Option import *
from Model.Server import *
from Model.ServiceProvider import  *
from Generator.Generator import *
import pandas as pd
import random_job
import numpy.random as Random


class GeneratorForModelAlibaba(Generator):
    """
    This inherits Generator and behave as a visitor for the NetworkProvider class. It's a bit tricky but it guarantees
    the same values generated for the MIP model
    """
    def _generateAvailableResources(self):
        np = NetworkProvider().getInstance()
        [np.addServer(Server(self.resources[0].generate(), self.resources[1].generate())) for _ in range(self.nbServers)]

    def _generateOptions(self):
        np = NetworkProvider().getInstance()
        for sp in np.getServiceProviders():
            for i in range(self.options.generate()):
                sp.addOption(Option(sp))

    def _generateBandwidthSaving(self):
        np = NetworkProvider().getInstance()
        totalResources = np.getTotalResources()
        for sp in np.getServiceProviders():
            for opt in sp.getOptions():
                resources = opt.getTotalResources()
                opt.setBandwidthSaving(self.bandwidth.generate(resources, totalResources))

    def _generateContainers(self):
        np = NetworkProvider().getInstance()
        for sp in np.getServiceProviders():
            for opt in sp.getOptions():
                rjob = random_job.random_job()
                for task in rjob.itertuples():
                    opt.addContainer(Container(task.CPU*100, task.memory*100))

    def _generateServiceProviders(self):
        np = NetworkProvider().getInstance()
        [np.addServiceProvider(ServiceProvider(self.execution_time.generate() if self.execution_time else None)) for _ in range(self.nbServiceProviders)]

    def generate(self, service_providers=True):
        NetworkProvider().getInstance().clean()
        #self._generateServiceProviders()
        self._generateAvailableResources()
        #self._generateOptions()
        #self._generateContainers()
        #self._generateRequiredResources()
        if service_providers:
            self.generateServiceProviders()
        #self._generateBandwidthSaving()

    def generateServiceProviders(self):
        print("Generating Service providers")
        np = NetworkProvider().getInstance()
        totalResources = np.getTotalResources()
        #while(self._K * np.getTotalResources()[0] > np.getSumAverageRequiredResources()[0] or
        # while k_ram < 1.8 || k_cpu < 1.8
        # k_ram = k_cpu

        container_meta = pd.read_csv("datasets-alibaba/container_meta.csv", header=None)
        container_usage = pd.read_csv("datasets-alibaba/container_usage_5mins.csv", header=None)
        print("CSVs read")
        apps = list(container_meta.loc[:,3].unique())

        print("Entering the loop")
        for _ in range(self.serviceProviders.generate()):
        #    self._K * np.getTotalResources()[1] > np.getSumAverageRequiredResources()[1]):
            sp = np.addServiceProvider(ServiceProvider(self.execution_time.generate() if self.execution_time else None))
            for _ in range(self.options.generate()):
                print("Generating option")

                notFound = True

                opt = sp.addOption(Option(sp))
                while (notFound):

                    app_du = apps.pop(Random.randint(0, len(apps)))

                    containers = container_meta.loc[container_meta[3] == app_du]

                    containers_ids = list(containers.loc[:,0].unique())

                    #rjob = random_job.random_job()
                    bw = 0
                    for container in containers_ids:
                        print("Inserting container")
                        ctmp = containers.loc[containers[0] == container].head(1)
                        ctmp2 = container_usage[container_usage[0] == container]
                        if len(ctmp2) == 0:
                            continue
                        notFound = False
                        opt.addContainer(Container(#int(ctmp2[3].mean()), int(ctmp2[4].mean())))
                            int(ctmp[5]),
                            int(ctmp[7])))
                        bw += ctmp2[8].mean() + ctmp2[9].mean()
                    opt.setBandwidthSaving(bw)

        avgZ = 0
        for sp in np.getServiceProviders():
            for opt in sp.getOptions():
                avgZ += len(opt.getContainers())
        avgZ /= len(np.getServiceProviders()) * len(np.getServiceProviders()[0].getOptions())

        self._K = \
            (
                np.getSumAverageRequiredResources()[0]/np.getTotalResources()[0] * len(np.getServiceProviders()) * avgZ,
                np.getSumAverageRequiredResources()[1]/np.getTotalResources()[1] * len(np.getServiceProviders()) * avgZ
            )

    def getK(self):
        return self._K

    def save_to_csv(self, suffix=""):
        np = NetworkProvider().getInstance()
        df = pd.DataFrame(columns=["sp", "opt", "container", "cpu", "ram", "cpu_tot_opt", "ram_tot_opt", "utility_opt"])
        for i, sp in enumerate(np.getServiceProviders()):
            for j, opt in enumerate(sp.getOptions()):
                for k, container in enumerate(opt.getContainers()):
                    df.loc[len(df)] = {
                        "sp": i,
                        "opt": j,
                        "container": k,
                        "cpu": container.getCpuReq(),
                        "ram": container.getRamReq(),
                        "cpu_tot_opt": opt.getCpuReq(),
                        "ram_tot_opt": opt.getRamReq(),
                        "utility_opt": opt.getBandwidthSaving()
                    }
        df.to_csv("results/google_traces_scenario_%d_sp_%d_opt_%s.csv" %
                  (len(np.getServiceProviders()), len(np.getServiceProviders()[0].getOptions()), suffix))

    def save_for_ilp(self, options_slice=1):
        np = NetworkProvider().getInstance()
        f = open("scenario.dat", "w+")
        f.write(f"param K := {self._K};\n")
        f.write(f"param nbServers := {len(np.getServers())};\n")
        f.write(f"param nbResources := 2;\n")
        f.write(f"param nbServiceProviders := {len(np.getServiceProviders())};\n")
        f.write("param availableResources :=")
        availableResources = {}
        for i, server in enumerate(np.getServers()):
            availableResources[i,0] = server.getTotalCpu()
            availableResources[i,1] = server.getTotalRam()
        f.write(self._printDict(availableResources))
        f.write("\n;\nparam nbOptions :=")
        options = []
        for i, sp in enumerate(np.getServiceProviders()):
            options.append(len(sp.getOptions()[0:options_slice]))
        f.write(self._printArray(options))
        f.write("\n;\nparam bandwidthSaving :=")
        bandwidthSaving = {}
        for i, sp in enumerate(np.getServiceProviders()):
            for j, opt in enumerate(sp.getOptions()[0:options_slice]):
                bandwidthSaving[i,j] = opt.getBandwidthSaving()
        f.write(self._printDict(bandwidthSaving))
        f.write("\n;\nparam nbContainers :=")
        containers = {}
        for i, sp in enumerate(np.getServiceProviders()):
            for j, opt in enumerate(sp.getOptions()[0:options_slice]):
                containers[i,j] = len(opt.getContainers())
        f.write(self._printDict(containers))
        f.write("\n;\nparam requiredResources :=")
        requiredResources = {}
        for i, sp in enumerate(np.getServiceProviders()):
            for j, opt in enumerate(sp.getOptions()[0:options_slice]):
                for k, container in enumerate(opt.getContainers()):
                    requiredResources[i,j,k,0] = container.getCpuReq()
                    requiredResources[i,j,k,1] = container.getRamReq()
        f.write(self._printDict(requiredResources))
        f.write("\n;\n\r")
        f.close()

