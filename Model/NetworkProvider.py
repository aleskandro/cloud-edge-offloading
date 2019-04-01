import operator

class NetworkProvider:
    class __NetworkProvider:
        def __init__(self):
            self.__servers = []
            self.__serviceProviders = []

        def addServiceProvider(self, serviceProvider):
            self.__serviceProviders.append(serviceProvider)

        def addServer(self, server):
            self.__servers.append(server)

        def __getBestHost(self, cpu, ram):
            bestHost = None
            for server in self.__servers:
                if server.getAvailableRam() < ram or server.getAvailableCpu < cpu:
                    continue
                if not bestHost or bestHost.getResidualValue() < server.getResidualValue():
                    bestHost = server
            return bestHost

        def makePlacement(self): # TODO
            convergence = False
            while (not convergence):
                for sp in self.__serviceProviders: # TODO multithread
                    sp.computeEfficiencies()

                # Set the convergence to true by default
                convergence = True

                for sp in self.__serviceProviders: # Cannot be multithreaded because of placement of containers onto servers
                    opt = sp.getMostEfficientOption()
                    placement = True
                    for container in opt.getContainers():
                        host = self.__getBestHost(container.getCpuReq(), container.getRamReq())
                        if host:
                            host.placeContainer(container)
                        else:
                            placement = False
                    # Discard the request if any container was not placed in the cluster
                    if not placement:
                        for container in opt.getContainers():
                            if container.getServer():
                                container.getServer().unplaceContainer(container)
                        # Make the convergence to False if in the previous step there was a default option set for this sp
                        if sp.getDefaultOption():
                            convergence = False
                        sp.setDefaultOption(None)
                    else:
                        # Verify if the choosen option is the same as the previous step
                        if not sp.getDefaultOption() is opt:
                            convergence = False

        def getServiceProviders(self):
            return self.__serviceProviders

        def getServers(self):
            return self.__servers
        
        def getTotalResources(self):
            ret = (0, 0)
            for server in self.__servers:
                ret = tuple(map(operator.add, ret, (server.getTotalCpu(), server.getTotalRam())))
            return ret
    instance = None # Can this be a private member?
    def getInstance(self):
        if not NetworkProvider.instance:
            NetworkProvider.instance = self.__NetworkProvider()
        return NetworkProvider.instance

    def __init__(self):
        self.getInstance()

