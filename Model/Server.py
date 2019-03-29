class Server:
    def __init__(self, totalCpu, totalRam):
        self.__totalCpu = totalCpu
        self.__totalRam = totalRam
        self.__occupiedRam = 0
        self.__occupiedCpu = 0

    def getTotalRam(self):
        return self.__totalRam

    def getAvailableRam(self):
        return self.__totalRam - self.__occupiedRam

    def getTotalCpu(self):
        return self.__totalCpu

    def getAvailableCpu(self):
        return self.__totalCpu - self.__occupiedCpu

    def getResidualValue(self):
        return self.getAvailableRam() * self.getAvailableCpu()

    def placeContainer(self, container):
        if (container.getCpuReq() > self.getAvailableCpu() or container.getRamReq() > self.getAvailableRam()):
            return False
        self.__occupiedRam += container.getRamReq()
        self.__occupiedCpu += container.getCpuReq()
        container.setServer(self)
        return True

    def unplaceContainer(self, container):
        if (not container.getServer() is self):
            return False
        self.__occupiedRam -= container.getRamReq()
        self.__occupiedCpu -= container.getCpuReq()
        container.setServer(None)
        return True



