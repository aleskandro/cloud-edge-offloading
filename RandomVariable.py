import math
#import numpy.random as Random
class RandomVariable:
    def __init__(self, val1, val2 = None):
        self._val1 = val1
        self._val2 = val2

    def generate(self):
        pass
    #def generateUniform(self):
    #    #return round(Random.uniform(self.avg - math.sqrt(3) * self.std, self.avg + math.sqrt(3) * self.std))
    #    return round(Random.uniform(self.val1, self.val2))

    #def generateNormal(self):
    #    return round(Random.normal(val1, val2))
