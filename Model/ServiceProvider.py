import multiprocessing
import operator
class ServiceProvider:
    def __init__(self, execution_time=None):
        self.__options = []
        self.__defaultOption = None
        self.__start_time = None
        self.__execution_time = execution_time

    def get_start_time(self):
        return self.__start_time

    def set_start_time(self, start_time):
        self.__start_time = start_time

    def get_execution_time(self):
        return self.__execution_time

    def set_execution_time(self, execution_time):
        self.__execution_time = execution_time

    def getOptions(self):
        return self.__options

    def addOption(self, option):
        self.__options.append(option)
        return option

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

    def __getCandidateOptions(self, options=None):
        subset = []
        opts = self.__options if options is None else self.__options[0:options]
        for opt in opts:
            bs = self.__defaultOption.getBandwidthSaving() if self.__defaultOption else 0
            if opt.getBandwidthSaving() > bs:
                subset.append(opt)
                opt.computeEfficiency()
        return subset

    def getBestOption(self, options=None):
        subset = self.__getCandidateOptions(options=options)
        if len(subset) == 0: return None
        return max(subset, key=lambda opt: opt.getEfficiency())

    def getMaxBsOption(self):
        return max(self.__options, key=lambda x: x.getBandwidthSaving())

    def __str__(self):
        return "Service provider: \n" + \
               "Default option: %s\nOptions: %s\n" % (str(self.__defaultOption), [str(item) for item in self.__options])
