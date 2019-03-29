import numpy.random as Random
from Random.RandomVariable import RandomVariable

class UniformRandomVariable(RandomVariable):
    def generate(self):
        ret = round(Random.uniform(self._val1, self._val2))
        while(ret < 0):
            ret = self.generate()
        return ret

