class Generator:
    def __init__(self, servers, serviceProviders, options, containers, resources, bandwidth, reqResources, execution_time = None, K = 0):
        self.servers = servers
        self.serviceProviders = serviceProviders
        self.options = options
        self.containers = containers
        self.resources = resources
        self.bandwidth = bandwidth
        self.reqResources = reqResources
        self.nbServers = self.servers.generate()
        self.nbResources = len(self.resources)
        self.nbServiceProviders = self.serviceProviders.generate()
        self.execution_time = execution_time
        self._K = K

    def _generateAvailableResources(self):
        availableResources = {}
        for i in range(round(self.nbServers)):
            for j in range(self.nbResources):
                availableResources[i,j] = self.resources[j].generate()
        return availableResources

    def _generateOptions(self):
        options = []
        for i in range(round(self.nbServiceProviders)):
            options.append(self.options.generate())
        return options

    def _generateBandwidthSaving(self, optionsArray):
        bandwidthSaving = {}
        for i in range(round(self.nbServiceProviders)):
            for j in range(optionsArray[i]):
                bandwidthSaving[i,j] = self.bandwidth.generate()
        return bandwidthSaving

    def _generateContainers(self, optionsArray):
        containers = {}
        for i in range(round(self.nbServiceProviders)):
            for j in range(round(optionsArray[i])):
                containers[i,j] = round(self.containers.generate())
        return containers

    def _generateRequiredResources(self, optionsArray, containersArray): 
        requiredResources = {}
        for i in range(round(self.nbServiceProviders)):
            for j in range(round(optionsArray[i])):
                for k in range(containersArray[i,j]):
                    for l in range(self.nbResources):
                        requiredResources[i,j,k,l] = self.reqResources[l].generate()
        return requiredResources

    def _printDict(self, theDict):
        string = ""
        for key, value in theDict.items():
            string += "\n"
            for index in key:
                string += str(index + 1) + " "
            string += str(value)
        return string

    def _printArray(self, theArray):
        string = ""
        for i in range(len(theArray)):
            string += "\n" + str(i + 1) + " " + str(theArray[i])
        return string

    def generate(self):
        availableResources = self._generateAvailableResources()
        options = self._generateOptions()
        containers = self._generateContainers(options)
        requiredResources = self._generateRequiredResources(options, containers)
        bandwidthSaving = self._generateBandwidthSaving(options)
        self._writeToFile(availableResources, options, containers, requiredResources, bandwidthSaving)

    def _writeToFile(self, availableResources, options, containers, requiredResources, bandwidthSaving):
        f = open("scenario.dat", "w+")
        f.write(f"param K := {self._K};\n")
        f.write(f"param nbServers := {self.nbServers};\n")
        f.write(f"param nbResources := {self.nbResources};\n")
        f.write(f"param nbServiceProviders := {self.nbServiceProviders};\n")
        f.write("param availableResources :=")
        f.write(self._printDict(availableResources))
        f.write("\n;\nparam nbOptions :=")
        f.write(self._printArray(options))
        f.write("\n;\nparam bandwidthSaving :=")
        f.write(self._printDict(bandwidthSaving))
        f.write("\n;\nparam nbContainers :=")
        f.write(self._printDict(containers))
        f.write("\n;\nparam requiredResources :=")
        f.write(self._printDict(requiredResources))
        f.write("\n;\n\r")
        f.close()


