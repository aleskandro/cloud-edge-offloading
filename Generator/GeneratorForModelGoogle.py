from Model.NetworkProvider import *
from Model.Container import *
from Model.Option import *
from Model.Server import *
from Model.ServiceProvider import  *
from Generator.Generator import *
import random_job


class GeneratorForModelGoogle(Generator):
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

    def generate(self):
        NetworkProvider().getInstance().clean()
        #self._generateServiceProviders()
        self._generateAvailableResources()
        #self._generateOptions()
        #self._generateContainers()
        #self._generateRequiredResources()
        self.generateServiceProviders()
        self._generateBandwidthSaving()

    def generateServiceProviders(self):
        np = NetworkProvider().getInstance()
        totalResources = np.getTotalResources()
        #while(self._K * np.getTotalResources()[0] > np.getSumAverageRequiredResources()[0] or
        for _ in range(self.serviceProviders.generate()):
        #    self._K * np.getTotalResources()[1] > np.getSumAverageRequiredResources()[1]):
            sp = np.addServiceProvider(ServiceProvider(self.execution_time.generate() if self.execution_time else None))
            for _ in range(self.options.generate()):
                opt = sp.addOption(Option(sp))
                rjob = random_job.random_job()
                for task in rjob.itertuples():
                    opt.addContainer(Container(task.CPU, task.memory))
                resources = opt.getTotalResources()
                opt.setBandwidthSaving(self.bandwidth.generate(resources, totalResources))
        self._K = (np.getSumAverageRequiredResources()[0]/np.getTotalResources()[0] + np.getSumAverageRequiredResources()[1]/np.getTotalResources()[1])/2

    def getK(self):
        return self._K
