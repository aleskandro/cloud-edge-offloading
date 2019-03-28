from Generator import Generator

class GeneratorBwConcave(Generator):
    def _generateBandwidthSaving(self, optionsArray, containersArray, requiredResources, availableResources):
        bandwidthSaving = {}
        totalResources  = {}
        for i in range(self.nbServers):
            for j in range(self.nbResources):
                if j in totalResources:
                    totalResources[j] += availableResources[i,j]
                else:
                    totalResources[j] = availableResources[i,j]

        for i in range(self.nbServiceProviders):
            for j in range(optionsArray[i]):
                resources = {}
                for k in range(containersArray[i,j]):
                    for l in range(self.nbResources):
                        if l in resources:
                            resources[l] += requiredResources[i, j, k, l]
                        else:
                            resources[l] = requiredResources[i, j, k, l]
                bandwidthSaving[i,j] = self.bandwidth.generate(resources, totalResources)
        return bandwidthSaving

    def generate(self):
        availableResources = self._generateAvailableResources()
        options = self._generateOptions()
        containers = self._generateContainers(options)
        requiredResources = self._generateRequiredResources(options, containers)
        bandwidthSaving = self._generateBandwidthSaving(options, containers, requiredResources, availableResources)
        self._writeToFile(availableResources, options, containers, requiredResources, bandwidthSaving)


