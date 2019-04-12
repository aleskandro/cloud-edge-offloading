import numpy.random as Random
from Random.RandomVariable import RandomVariable

class PoissonRandomVariable(RandomVariable):
    def generate(self):
        ret = round(Random.poisson(self._val1))
        while(ret < 0):
            ret = self.generate()
        return ret

