from Model.NetworkProvider import *
from Model.Container import *
from Model.Option import *
from Model.Server import *
from Model.ServiceProvider import  *
from Generator.Generator import *
import pandas as pd

class GeneratorForModel(Generator):
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
                [opt.addContainer(Container(0,0)) for _ in range(self.containers.generate())]

    def _generateRequiredResources(self):
        np = NetworkProvider().getInstance()
        for sp in np.getServiceProviders():
            for opt in sp.getOptions():
                for ct in opt.getContainers():
                    ct.setCpuReq(self.reqResources[0].generate())
                    ct.setRamReq(self.reqResources[1].generate())

    def _generateServiceProviders(self):
        np = NetworkProvider().getInstance()
        [np.addServiceProvider(ServiceProvider(self.execution_time.generate() if self.execution_time else None)) for _ in range(self.nbServiceProviders)]

    def generate(self):
        NetworkProvider().getInstance().clean()
        self._generateServiceProviders()
        self._generateAvailableResources()
        self._generateOptions()
        self._generateContainers()
        self._generateRequiredResources()
        self._generateBandwidthSaving()

    def generateServiceProviders(self):
        np = NetworkProvider().getInstance()
        totalResources = np.getTotalResources()
        for _ in range(self.serviceProviders.generate()):
            sp = np.addServiceProvider(ServiceProvider(self.execution_time.generate() if self.execution_time else None))
            for _ in range(self.options.generate()):
                opt = sp.addOption(Option(sp))
                for _ in range(self.containers.generate()):
                    opt.addContainer(Container(self.reqResources[0].generate(), self.reqResources[1].generate()))
                resources = opt.getTotalResources()
                opt.setBandwidthSaving(self.bandwidth.generate(resources, totalResources))

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