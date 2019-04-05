import multiprocessing
import operator
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

    def __getCandidateOptions(self):
        subset = []
        for opt in self.__options:
            bs = self.__defaultOption.getBandwidthSaving() if self.__defaultOption else 0
            if opt.getBandwidthSaving() > bs:
                subset.append(opt)
                opt.computeEfficiency()
        return subset

    def getBestOption(self):
        subset = self.__getCandidateOptions()
        if len(subset) == 0: return None
        return max(subset, key=lambda opt: opt.getEfficiency())
