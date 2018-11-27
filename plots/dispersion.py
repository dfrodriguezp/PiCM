import numpy
from matplotlib import pyplot
from params import *

def nextpow2(A):
    if type(A) == int or type(A) == float or type(A) == numpy.int64 or type(A) == numpy.float64:
        p = 0
        while True:
            if (2 ** p >= numpy.abs(A)):
                return p
            p += 1
    else:
        B = list()
        for i in A:
            B.append(nextpow2(i))
        B = numpy.array(B)
        return B

# x, E = numpy.loadtxt("results/field/step_80.dat", unpack=True)

# pyplot.figure()
# pyplot.plot(x, E, "-o")
# pyplot.show()

matrix = list()
for step in range(steps):
    if step % 10 == 0:
        x, E = numpy.loadtxt("results/field/step_{}.dat".format(step), unpack=True)
        matrix.append(E)

matrix = numpy.array(matrix)
matrix = numpy.fft.fft2(matrix)
# matrix = numpy.abs(matrix)

w = numpy.arange(len(matrix))
# w = nextpow2(w)
# w = numpy.fft.fftshift(w)

pyplot.figure()
pyplot.contourf(range(NG), w, matrix)
# pyplot.contourf(final_data)
pyplot.colorbar(ax=pyplot.gca())
pyplot.xlim(0, 2)
pyplot.ylim(0, 6)
pyplot.xlabel(r"$k$", fontsize=20)
pyplot.ylabel(r"$\omega$", fontsize=20)
# pyplot.show()
pyplot.savefig("fftonly_bounded.pdf")