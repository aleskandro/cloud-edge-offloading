from Model.NetworkProvider import *
class GeneratorForModel(Generator):
    def _generateAvailableResources(self):
        np = NetworkProvider().getInstance()
        [np.addServer(self.resources[0].generate(), self.resources[1].generate()) for i in range(self.nbServers)]

    def _generateOptions(self):
        np = NetworkProvider().getInstance()
        for sp in np.getServiceProviders():
            for i in range(self.options.generate()):
                sp.addOption(Option(sp))

    def _generateBandwidthSaving(self):
        np = NetworkProvider().getInstance()
        totalResources = np.getTotalResources()
        for sp in np.getServiceProviders():
            for opt in sp.getOptions():
                resources = opt.getTotalResources()
                opt.setBandwidthSaving(self.bandwidth.generate(resources, totalResources)

    def _generateContainers(self):
        np = NetworkProvider().getInstance()
        for sp in np.getServiceProviders():
            for opt in sp.getOptions():
                [opt.addContainer(Container(0,0)) for _ in range(self.containers.generate())]

    def _generateRequiredResources(self):
        np = NetworkProvider().getInstance()
        for sp in np.getServiceProviders():
            for opt in sp.getOptions():
                for ct in opt.getContainers():
                    ct.setCpuReq(self.reqResources[0].generate())
                    ct.setRamReq(self.reqResources[1].generate())

    def generate(self):
        availableResources = self._generateAvailableResources()
        options = self._generateOptions()
        containers = self._generateContainers(options)
        requiredResources = self._generateRequiredResources(options, containers)
        bandwidthSaving = self._generateBandwidthSaving(options)

