import numpy.random as Random
from Random.RandomVariable import RandomVariable

class NormalRandomVariable(RandomVariable):
    def generate(self):
        ret = round(Random.normal(self._val1, self._val2))
        while (ret < 0):
            ret = self.generate()
        return ret

