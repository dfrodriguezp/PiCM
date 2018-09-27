import numpy
from matplotlib import pyplot
from params import *

# x, E = numpy.loadtxt("results/field/step_100.dat", unpack=True)

# pyplot.figure()
# pyplot.plot(x, E, "-o")
# pyplot.show()

matrix = list()
for step in range(steps):
    if step % 10 == 0:
        x, E = numpy.loadtxt("results/field/step_{}.dat".format(step), unpack=True)
        matrix.append(E)

matrix = numpy.array(matrix)
matrix_ft = numpy.fft.fft2(matrix)
final_data = numpy.abs(matrix_ft)

pyplot.figure()
# pyplot.pcolormesh(final_data, cmap="jet")
pyplot.contour(final_data)
pyplot.colorbar(ax=pyplot.gca())
pyplot.xlim(0, 2)
# pyplot.ylim(0, 6)
pyplot.show()