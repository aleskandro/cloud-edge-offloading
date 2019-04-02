import multiprocessing

class ServiceProvider:
    def __init__(self):
        self.__options = []
        self.__defaultOption = None

    def getOptions(self):
        return self.__options

    def addOption(self, option):
        self.__options.append(option)

    def getDefaultOption(self):
        return self.__defaultOption

    def setDefaultOption(self, option):
        self.__defaultOption = option

    def getMostEfficientOption(self):
        mostEfficientOption = self.__defaultOption
        for option in self.__options:
            efficiency = 0 if not mostEfficientOption else mostEfficientOption.getEfficiency()
            if (efficiency < option.getEfficiency()):
                mostEfficientOption = option
        return mostEfficientOption

    def ce(self, option):
        option.computeEfficiency()

    def computeEfficiencies(self):
        # TODO restore multiprocessing but with blocking/join
        #with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        #    p.map(self.ce, self.__options)
        for opt in self.__options:
            opt.computeEfficiency()
