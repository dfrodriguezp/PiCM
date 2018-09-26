import numpy
from matplotlib import pyplot
from params import *

step = 1500
x, v = numpy.loadtxt("results/phase_space/step_{}.dat".format(step), unpack=True)

pyplot.figure()
pyplot.scatter(x, v, s=2)
pyplot.xlim(0, L)
pyplot.show()
pyplot.close()