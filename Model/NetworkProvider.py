import operator

class NetworkProvider:
    class __NetworkProvider:
        def clean(self):
            self.__servers = []
            self.__serviceProviders = []

        def __init__(self):
            self.clean()

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

        def makePlacement(self):
            convergence = False
            datas = [[] for _ in self.__serviceProviders]
            while not convergence:
                for sp in self.__serviceProviders: # TODO multithread
                    sp.computeEfficiencies()

                # Set the convergence to true by default
                convergence = True
                sp_index = 0
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
                        datas[sp_index].append(sp.getOptions().index(sp.getDefaultOption()) + 1 if sp.getDefaultOption()
                                               else 0)
                    else:
                        # Verify if the choosen option is the same as the previous step
                        if not sp.getDefaultOption() is opt:
                            convergence = False
            # TODO print datas to len(sps) graphs
            self.__print_datas(datas)

        def __print_datas(self, datas):
            pass

        def getServiceProviders(self):
            return self.__serviceProviders

        def getServers(self):
            return self.__servers
        
        def getTotalResources(self):
            ret = (0, 0)
            for server in self.__servers:
                ret = tuple(map(operator.add, ret, (server.getTotalCpu(), server.getTotalRam())))
            return ret

    instance = None  # Can this be a private member?

    def getInstance(self):
        if not NetworkProvider.instance:
            NetworkProvider.instance = self.__NetworkProvider()
        return NetworkProvider.instance

    def __init__(self):
        self.getInstance()

