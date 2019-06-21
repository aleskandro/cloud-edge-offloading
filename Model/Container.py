class Container:
    def __init__(self, cpuReq, ramReq):
        self.__cpuReq = cpuReq
        self.__ramReq = ramReq
        self.__server = None
        self.__old_server = None

    def getCpuReq(self):
        return self.__cpuReq

    def getRamReq(self):
        return self.__ramReq

    def getServer(self):
        return self.__server

    def setServer(self, server):
        self.__server = server

    def set_old_server(self, server):
        self.__old_server = server

    def get_old_server(self):
        return self.__old_server

    def setRamReq(self, ram):
        self.__ramReq = ram

    def setCpuReq(self, cpu):
        self.__cpuReq = cpu

    def __str__(self):
        return "CPU: %f - RAM: %f" % (self.getCpuReq(), self.getRamReq())
