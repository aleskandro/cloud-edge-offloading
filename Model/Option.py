from Model.NetworkProvider import *
class Option:
    def __init__(self, serviceProvider):
        self.__containers = []
        self.__bandwidthSaving = 0
        self.__efficiency = 0
        self.__serviceProvider = serviceProvider

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

    def getTotalResources(self):
        return self.getCpuReq(), self.getRamReq()

    def setBandwidthSaving(self, bandwidthSaving):
        self.__bandwidthSaving = bandwidthSaving

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

    def computeEfficiency(self): # TODO multithreading as in unplaceContainers
        denominator = 0
        defaultSPCpuSum = 0
        defaultSPRamSum = 0
        for sp in NetworkProvider().getInstance().getServiceProviders():
            defaultSPCpuSum += sp.getDefaultOption().getCpuReq()
            defaultSPRamSum += sp.getDefaultOption().getRamReq()

        for server in NetworkProvider().getInstance().getServers():
           denominator += self.getCpuReq() * max((defaultSPCpuSum - server.getTotalCpu()), 0)
           denominator += self.getRamReq() * max((defaultSPRamSum - server.getTotalRam()), 0)

        self.__efficiency = self.__bandwidthSaving / denominator

