import numpy.random as Random
from RandomVariable import RandomVariable

class ResourceDependentRandomVariable(RandomVariable):
    def generate(self, resources, totalResources):
        ret = 0
        alpha = []
        sumAlpha = 0
        upperBound = 1 # TODO in constructor
        for i in range(len(resources) - 1):
            rn = Random.uniform(0, upperBound - sumAlpha)
            sumAlpha += rn
            alpha.append(rn)
        alpha.append(upperBound - sumAlpha)
        for i in range(len(resources)):
            ret += alpha[i]*(float(resources[i])/totalResources[i])**(1/self._val1.generate()) ## TODO self._val1 as array in resources

        while (ret < 0):
            ret = self.generate()
        return ret

