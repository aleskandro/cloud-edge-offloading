import multiprocessing

class ServiceProvider:
    def __init__(self):
        self.__options = []
        self.__defaultOption = None

    def getOptions(self):
        return self.__options

    def addOption(self, option):
        if not self.__defaultOption:
            self.setDefaultOption(option)
        self.__options.append(option)

    def getDefaultOption(self):
        return self.__defaultOption

    def setDefaultOption(self, option):
        self.__defaultOption = option

    def getMostEfficientOption(self):
        mostEfficientOption = self.__defaultOption
        for option in self.__options:
            if (mostEfficientOption.getEfficiency() < option.getEfficiency()):
                mostEfficientOption = option
        return mostEfficientOption

    def __ce(self, option):
        option.computeEfficiency()

    def computeEfficiencies(self):
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            p.map(self.__ce, self.__options)

