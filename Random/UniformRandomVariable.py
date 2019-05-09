import numpy.random as Random
from Random.RandomVariable import RandomVariable

class UniformRandomVariable(RandomVariable):
    def generate(self, to_int=True):
        ret = Random.uniform(self._val1, self._val2)
        if self._to_int:
            ret = round(ret)
        while(ret < 0):
            ret = self.generate()
        return ret

