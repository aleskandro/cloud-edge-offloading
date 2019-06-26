from Model.NetworkProvider import *
class Option:
    def __init__(self, serviceProvider):
        self.__containers = []
        self.__bandwidthSaving = 0
        self.__efficiency = 0
        self.__serviceProvider = serviceProvider
        self.__tried = False

    def tried(self, tried=True):
        self.__tried = tried

    def wasTried(self):
        return self.__tried


    def getCpuReq(self): # TODO multithread
        cpu = 0
        for container in self.__containers:
            cpu += container.getCpuReq()
        return cpu

    def getRamReq(self): # TODO multithread
        ram = 0
        for container in self.__containers:
            ram += container.getRamReq()
        return ram

    def getServiceProvider(self):
        return self.__serviceProvider

    def getTotalResources(self):
        return self.getCpuReq(), self.getRamReq()

    def setBandwidthSaving(self, bandwidthSaving):
        self.__bandwidthSaving = bandwidthSaving

    def getBandwidthSaving(self):
        return self.__bandwidthSaving

    def addContainer(self, container):
        self.__containers.append(container)

    def getContainers(self):
        return self.__containers

    def unplaceContainers(self):
        # TODO multithread, unplaceContainer in server is not safe because of non-blocking sums https://pypi.org/project/atomiclong/
        for container in self.__containers:
            if (container.getServer()):
                container.getServer().unplaceContainer(container)

    def getEfficiency(self):
        return self.__efficiency

#    def computeEfficiency(self): # TODO multithreading as in unplaceContainers
#        denominator = 0
#        defaultSPCpuSum = 0
#        defaultSPRamSum = 0
#        for sp in NetworkProvider().getInstance().getServiceProviders():
#            if sp.getDefaultOption():
#                defaultSPCpuSum += sp.getDefaultOption().getCpuReq()
#                defaultSPRamSum += sp.getDefaultOption().getRamReq()
#
#        for server in NetworkProvider().getInstance().getServers():
#           denominator += self.getCpuReq() * max((defaultSPCpuSum - server.getTotalCpu()), 1) # TODO restore to 0
#           denominator += self.getRamReq() * max((defaultSPRamSum - server.getTotalRam()), 1)
#        self.__efficiency = self.__bandwidthSaving / denominator

    def computeEfficiency(self):
        # Return 0 if computing efficiency to jump to itself
        if (self is self.getServiceProvider().getDefaultOption()):
            return 0
        denominator = 0
        numerator = self.getBandwidthSaving()
        numerator -= self.getServiceProvider().getDefaultOption().getBandwidthSaving() if \
            self.getServiceProvider().getDefaultOption() else 0

        h_cpu, h_ram = NetworkProvider().getInstance().get_h_l()
        tcpu = self.getServiceProvider().getDefaultOption().getCpuReq() if self.getServiceProvider().getDefaultOption() else 0
        tram = self.getServiceProvider().getDefaultOption().getRamReq() if self.getServiceProvider().getDefaultOption() else 0
        denominator += (self.getCpuReq() - tcpu) * h_cpu
        denominator += (self.getRamReq() - tram) * h_ram
        self.__efficiency = numerator / denominator

    def getResourcesIncreasing(self):
        tcpu = self.getServiceProvider().getDefaultOption().getCpuReq() if self.getServiceProvider().getDefaultOption() else 0
        tram = self.getServiceProvider().getDefaultOption().getRamReq() if self.getServiceProvider().getDefaultOption() else 0
        return (self.getCpuReq() - tcpu), (self.getRamReq() - tram)

    def __str__(self):
        return "Option \n" + \
            "Bandwidth saving: %f\nContainers: \n %s" % (self.__bandwidthSaving, [str(item) for item in self.__containers])
