import operator
import matplotlib.pyplot as plt
import numpy.random as Random
import math
import pandas as pd

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

        def getBestHost(self, cpu, ram):
            bestHost = None
            for server in self.__servers:
                if server.getAvailableRam() < ram or server.getAvailableCpu() < cpu:
                    continue
                if not bestHost or bestHost.getResidualValue() < server.getResidualValue():
                    bestHost = server
            return bestHost

        def getBestHostMin(self, cpu, ram):
            bestHost = None
            for server in self.__servers:
                if server.getAvailableRam() < ram or server.getAvailableCpu() < cpu:
                    continue
                if not bestHost or bestHost.getResidualValue() > server.getResidualValue():
                    bestHost = server
            return bestHost

        def getBestHostScalarProduct(self, cpu, ram):
            bestHost = None
            containerVector = (cpu, ram)
            projection = 0
            for server in self.__servers:
                if server.getAvailableRam() < ram or server.getAvailableCpu() < cpu:
                    continue

                desiredVector = (server.getAvailableCpu(), server.getAvailableRam())
                desiredVectorNorm = math.sqrt(server.getAvailableCpu()**2 + server.getAvailableRam()**2)
                desiredVector = map(lambda x: x/desiredVectorNorm, desiredVector)
                projection_new = sum(p*q for p,q in zip(desiredVector, containerVector))

                if not bestHost or projection_new > projection:
                    bestHost = server
                    projection = projection_new

            return bestHost

        def getBandwidthSaving(self):
            ret = 0
            for sp in self.__serviceProviders:
                ret += sp.getDefaultOption().getBandwidthSaving() if sp.getDefaultOption() else 0
            return ret

        def getRelativeBandwidthSaving(self):
            ret = self.getBandwidthSaving()
            max_bs = 0
            for sp in self.__serviceProviders:
                max_bs += sp.getMaxBsOption().getBandwidthSaving()
            #return ret / max_bs
            return ret / len(self.__serviceProviders)

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
                    option.tried(False)
                    for container in option.getContainers():
                        if container.getServer():
                            container.getServer().unplaceContainer(container)
                sp.setDefaultOption(None)
            # Cluster clean
        def clean_cluster(self):
            self.__clean_cluster()

        def get_h_l(self):
            h_cpu = 0
            h_ram = 0
            for server in self.getServers():
                h_cpu += server.getTotalCpu()
                h_ram += server.getTotalRam()
            h_cpu = 1/h_cpu
            h_ram = 1/h_ram
            #return h_cpu, h_ram
            return 5, 1

        def makePlacement(self, placement_id, time=0, options_slice=None, get_best_host=None, collect_iterations_report=False):
            iterations_report = pd.DataFrame(columns=["Iteration", "Utility", "ExpectedUtility", "BestJumpEfficiency"])
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
            iteration = 0
            old_option = None
            while(fitting): #and limit > 0):
                limit-=1
                options = []
                for sp in self.__serviceProviders:
                    opt = sp.getBestOption(options_slice)
                    if opt:
                        options.append(opt)
                if len(options) == 0:
                    break
                candidateOption = max(options, key=lambda x: x.getEfficiency())
                # Unplace old containers if options for the selected sp has changed
                #option_has_changed = candidateOption.getServiceProvider().getDefaultOption() != candidateOption
                placement = True
                #if option_has_changed:
                #    # Unplace old option from the cluster
                if candidateOption.getServiceProvider().getDefaultOption():
                     for container in candidateOption.getServiceProvider().getDefaultOption().getContainers():
                         #if container.getServer():
                         container.getServer().unplaceContainer(container)

                # Try to place new option on the cluster
                for container in candidateOption.getContainers():
                    host = None
                    if get_best_host is None:
                        host = self.getBestHost(container.getCpuReq(), container.getRamReq())
                    else:
                        host = get_best_host(container.getCpuReq(), container.getRamReq())
                    if host:
                        host.placeContainer(container)
                    else:
                        placement = False
                # Discard the request if any container was not placed in the cluster

                if not placement:
                    for container in candidateOption.getContainers():
                        if container.getServer():
                            container.getServer().unplaceContainer(container)
                    # Restore old option
                    if candidateOption.getServiceProvider().getDefaultOption():
                        for container in candidateOption.getServiceProvider().getDefaultOption().getContainers():
                            host = container.get_old_server()
                            if host is None:
                                print("WARNING: Host is none for old option")
                            if host:
                                host.placeContainer(container)
                    # Algorithm end
                    #fitting = False
                    # Algorithm not anymore end here but option will be set to tried
                    # The algorithm now end whenever the candidateOption is null
                    candidateOption.tried()
                else:
                    candidateOption.getServiceProvider().setDefaultOption(candidateOption)
                    candidateOption.getServiceProvider().set_start_time(time)  # Option is being deployed, setting start time

                for sp_index, sp in enumerate(self.__serviceProviders):
                    datas[sp_index].append(sp.getOptions().index(sp.getDefaultOption()) + 1 if sp.getDefaultOption()
                        else 0)

                if collect_iterations_report:

                    h_cpu, h_ram = self.get_h_l()
                    utility_expected = 0
                    if old_option is not None:
                        utility_expected = old_option.getEfficiency() * \
                            (
                                   h_cpu * (self.getTotalResources()[0] - old_option.getResourcesIncreasing()[0]) +
                                   h_ram * (self.getTotalResources()[1] - old_option.getResourcesIncreasing()[1])
                            )
                        if utility_expected < 0:
                            print("ALERT! negative utility expected")
                        utility_expected += self.getBandwidthSaving()
                        utility_expected = min(utility_expected, float(iterations_report.tail(1)["ExpectedUtility"]))

                        #if utility_expected > len(self.__serviceProviders):
                        #    utility_expected = len(self.__serviceProviders)
                    if placement:
                        old_option = candidateOption
                    new_row = {"Iteration": iteration, "Utility": self.getBandwidthSaving(),
                               "ExpectedUtility": utility_expected, "BestJumpEfficiency": candidateOption.getEfficiency()}
                    for index, server in enumerate(self.getServers()):
                        new_row["%d_CPU" % index] = (server.getTotalCpu() - server.getAvailableCpu()) \
                                                    / server.getTotalCpu()
                        new_row["%d_RAM" % index] = (server.getTotalRam() - server.getAvailableRam()) \
                                                    / server.getTotalRam()
                    iterations_report = iterations_report.append(new_row, ignore_index=True)
                    iteration += 1
            #print(datas)
            return iterations_report

        def make_placement_naive(self, placement_id=0): # Not for temporal execution
            self.__clean_cluster()
            for sp in self.getServiceProviders():
                opt = sp.getOptions()[math.ceil(Random.uniform(0, len(sp.getOptions()) - 1))]
                sp.setDefaultOption(opt)
                for container in opt.getContainers():
                    # Try to place to any host
                    placed = False
                    for host in self.getServers():
                        if host.placeContainer(container):
                            placed = True
                            break
                    if not placed: # revert
                        for container in opt.getContainers():
                            if container.getServer():
                                container.getServer().unplaceContainer(container)
                        sp.setDefaultOption(None)
                        break



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

        def getSumAverageRequiredResources(self):
            ret = (0, 0)
            for sp in self.__serviceProviders:
                t = (0, 0)
                for opt in sp.getOptions():
                    t = tuple(map(operator.add, t, (opt.getCpuReq(), opt.getRamReq())))
                    t = tuple(map(operator.truediv, t, (len(opt.getContainers()), len(opt.getContainers()))))
                t = tuple(map(operator.truediv, t, (len(sp.getOptions()), len(sp.getOptions()))))
                ret = tuple(map(operator.add, ret, t))
            return tuple(map(operator.truediv, ret, (len(self.__serviceProviders), len(self.__serviceProviders))))

    instance = None  # Can this be a private member?

    def getInstance(self):
        if not NetworkProvider.instance:
            NetworkProvider.instance = self.__NetworkProvider()
        return NetworkProvider.instance

    def __init__(self):
        self.getInstance()

