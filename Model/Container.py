class Container:
    def __init__(self, cpuReq, ramReq):
        self.__cpuReq = cpuReq
        self.__ramReq = ramReq
        self.__server = None

    def getCpuReq(self):
        return self.__cpuReq

    def getRamReq(self):
        return self.__ramReq

    def getServer(self):
        return self.__server

    def setServer(self, server):
        self.__server = server
