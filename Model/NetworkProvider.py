import operator
import matplotlib.pyplot as plt


class NetworkProvider:
    class __NetworkProvider:
        def clean(self):
            self.__servers = []
            self.__serviceProviders = []

        def __init__(self):
            self.clean()

        def addServiceProvider(self, serviceProvider):
            self.__serviceProviders.append(serviceProvider)
            return serviceProvider

        def deleteServiceProvider(self, serviceProvider):
            if (serviceProvider.getDefaultOption()):
                for c in serviceProvider.getDefaultOption().getContainers():
                    if c.getServer():
                        c.getServer().unplaceContainer(c)
            self.__serviceProviders.remove(serviceProvider)

        def addServer(self, server):
            self.__servers.append(server)

        def __getBestHost(self, cpu, ram):
            bestHost = None
            for server in self.__servers:
                if server.getAvailableRam() < ram or server.getAvailableCpu() < cpu:
                    continue
                if not bestHost or bestHost.getResidualValue() < server.getResidualValue():
                    bestHost = server
            return bestHost

        def getBandwidthSaving(self):
            ret = 0
            for sp in self.__serviceProviders:
                ret += sp.getDefaultOption().getBandwidthSaving() if sp.getDefaultOption() else 0
            return ret

        def getRemainingResources(self):
            cpu = 0
            ram = 0
            cpu_available = 0
            ram_available = 0
            for server in self.__servers:
                cpu += server.getTotalCpu()
                ram += server.getTotalRam()

            for server in self.__servers:
                cpu_available += server.getAvailableCpu()
                ram_available += server.getAvailableRam()

            return cpu_available/cpu, ram_available/ram

        def __clean_cluster(self):
            # Clean the cluster
            for sp in self.__serviceProviders:
                for option in sp.getOptions():
                    for container in option.getContainers():
                        if container.getServer():
                            container.getServer().unplaceContainer(container)
            # Cluster clean

        def makePlacement(self, placement_id, time=0):
            fitting = True
            if time == 0:
                self.__clean_cluster() # Clean the cluster at time 0 or for a non time-batched execution
            else:
                for sp in self.__serviceProviders:
                    if not sp.get_start_time() is None and not sp.get_execution_time() is None:
                            #time - sp.get_start_time() >= sp.get_execution_time(): # if time elapsed for an execution job
                        self.deleteServiceProvider(sp)

            datas = [[] for _ in self.__serviceProviders]
            limit = 100 # limits the number of iteration to 100, for safety on not convergence but tricky, it should be the total number of options
            while(fitting and limit > 0):
                limit-=1
                options = []
                for sp in self.__serviceProviders:
                    opt = sp.getBestOption()
                    if opt:
                        options.append(opt)
                candidateOption = max(options, key=lambda x: x.getEfficiency()) if len(options) > 0 else None
                if not candidateOption:
                    break
                # Unplace old containers if options for the selected sp has changed
                #option_has_changed = candidateOption.getServiceProvider().getDefaultOption() != candidateOption
                placement = True
                #if option_has_changed:
                #    # Unplace old option from the cluster
                if candidateOption.getServiceProvider().getDefaultOption():
                     for container in candidateOption.getServiceProvider().getDefaultOption().getContainers():
                         if container.getServer():
                             container.getServer().unplaceContainer(container)
                # Try to place new option on the cluster
                for container in candidateOption.getContainers():
                    host = self.__getBestHost(container.getCpuReq(), container.getRamReq())
                    if host:
                        host.placeContainer(container)
                    else:
                        placement = False
                # Discard the request if any container was not placed in the cluster
                if not placement:
                    for container in candidateOption.getContainers():
                        if container.getServer():
                            container.getServer().unplaceContainer(container)
                    # Restore old option (Warning: maybe the same placement as the previous one is not guaranteed)
                    if candidateOption.getServiceProvider().getDefaultOption():
                        for container in candidateOption.getServiceProvider().getDefaultOption().getContainers():
                            host = self.__getBestHost(container.getCpuReq(), container.getRamReq())
                            if host:
                                host.placeContainer(container)
                    # Algorithm end
                    fitting = False
                else:
                    candidateOption.getServiceProvider().setDefaultOption(candidateOption)
                    candidateOption.getServiceProvider().set_start_time(time)  # Option is being deployed, setting start time

                sp_index = 0
                for sp in self.__serviceProviders:
                    datas[sp_index].append(sp.getOptions().index(sp.getDefaultOption()) + 1 if sp.getDefaultOption()
                        else 0)
                    sp_index += 1
            print(datas)
            #self.__print_datas(datas, placement_id)
#        def makePlacement(self, placement_id):
#            convergence = False
#            datas = [[] for _ in self.__serviceProviders]
#            print(datas)
#            print("Startgin placement")
#            limit = 100
#            while not convergence and limit > 0:
#                limit -= 1
#                print("Starting iteration...")
#                print("Computing efficiencies...")
#                for sp in self.__serviceProviders: # TODO multithread
#                    sp.computeEfficiencies()
#
#                # Set the convergence to true by default
#                convergence = True
#                sp_index = 0
#                self.__clean_cluster()
#                print("Trying to make the placement for each service provider")
#                for sp in self.__serviceProviders: # Cannot be multithreaded because of placement of containers onto servers
#                    old_is_none = not sp.getDefaultOption()
#                    opt = sp.getMostEfficientOption()
#                    placement = True
#                    for container in opt.getContainers():
#                        host = self.__getBestHost(container.getCpuReq(), container.getRamReq())
#                        if host:
#                            host.placeContainer(container)
#                        else:
#                            placement = False
#                    # Discard the request if any container was not placed in the cluster
#                    if not placement:
#                        for container in opt.getContainers():
#                            if container.getServer():
#                                container.getServer().unplaceContainer(container)
#                        # Make the convergence to False if in the previous step there was a default option set for this sp
#                        if sp.getDefaultOption():
#                            convergence = False
#                        sp.setDefaultOption(None)
#                    else:
#                        # Verify if the choosen option is the same as the previous step
#                        if not sp.getDefaultOption() is opt:
#                            convergence = False
#                        sp.setDefaultOption(opt)
#                    datas[sp_index].append(sp.getOptions().index(sp.getDefaultOption()) + 1 if sp.getDefaultOption()
#                                           else 0)
#                    sp_index += 1
#            print(datas)
#            self.__print_datas(datas, placement_id)

        def __print_datas(self, datas, placement_id):
            x = range(len(datas[0]))
            fig, axs = plt.subplots(nrows=1, ncols=1)
            [axs.plot(x, datas[i], label="SP %d" % i) for i in range(len(datas))]
            axs.legend(loc="best")
            fig.savefig("results/chosen-option-per-sp-%d.png" % placement_id)
            # TODO does a better way exist to assign id to the image?

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

