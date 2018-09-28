import sys
sys.path.append("/home/daniel/Dropbox/PIC_1D_python_n/PIC-1D")
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from params import *
import numpy


colors = numpy.array(["r"] * NP)
colors[1::2] = "b"

potential = PdfPages("potential.pdf")
density = PdfPages("density.pdf")
Efield = PdfPages("Efield.pdf")
phaseSpace = PdfPages("phase_space.pdf")

for step in tqdm(numpy.where(numpy.arange(steps) % 100 == 0)[0]):
    x, v = numpy.loadtxt("../results/phase_space/step_{}.dat".format(step), unpack=True)
    x_i, phi = numpy.loadtxt("../results/potential/step_{}.dat".format(step), unpack=True)
    rho = numpy.loadtxt("../results/density/step_{}.dat".format(step), usecols=(1,), unpack=True)
    E = numpy.loadtxt("../results/field/step_{}.dat".format(step), usecols=(1,), unpack=True)

    pyplot.figure()
    pyplot.title("step {}".format(step), fontsize=25)
    pyplot.scatter(x, v, s=2, color=colors)
    pyplot.xlim(0, L)
    pyplot.ylim(-6, 6)
    pyplot.xlabel("$x$", fontsize=20)
    pyplot.ylabel("$v$", fontsize=20)
    pyplot.grid()
    pyplot.savefig(phaseSpace, format="pdf")

    pyplot.figure()
    pyplot.title("step {}".format(step), fontsize=25)
    pyplot.plot(x_i, phi, "-o", color="r")
    pyplot.xlim(0, L)
    pyplot.ylim(-1.5, 1.5)
    pyplot.xlabel("$x$", fontsize=20)
    pyplot.ylabel(r"$\phi$", fontsize=20)
    pyplot.grid()
    pyplot.savefig(potential, format="pdf")

    pyplot.figure()
    pyplot.title("step {}".format(step), fontsize=25)
    pyplot.plot(x_i, rho, "-o")
    pyplot.xlim(0, L)
    pyplot.ylim(-0.8, 0.8)
    pyplot.xlabel("$x$", fontsize=20)
    pyplot.ylabel(r"$\rho$", fontsize=20)
    pyplot.grid()
    pyplot.savefig(density, format="pdf")

    pyplot.figure()
    pyplot.title("step {}".format(step), fontsize=25)
    pyplot.plot(x_i, E, "-o", color="g")
    pyplot.xlim(0, L)
    pyplot.ylim(-0.8, 0.8)
    pyplot.xlabel("$x$", fontsize=20)
    pyplot.ylabel(r"$E$", fontsize=20)
    pyplot.grid()
    pyplot.savefig(Efield, format="pdf")

density.close()
Efield.close()
potential.close()
phaseSpace.close()


KE, FE, step = numpy.loadtxt("../results/energy/energy.dat", unpack=True)

pyplot.figure()
pyplot.plot(step, KE, "--o", ms=1)
pyplot.xlim(0, steps)
pyplot.xlabel("steps", fontsize=20)
pyplot.ylabel("$K$", fontsize=20)
pyplot.grid()
pyplot.savefig("kinetic_energy.pdf")

pyplot.figure()
pyplot.plot(step, FE, "--o", ms=1)
pyplot.xlim(0, steps)
pyplot.xlabel("steps", fontsize=20)
pyplot.ylabel(r"$E_{F}$", fontsize=20)
pyplot.grid()
pyplot.savefig("field_energy.pdf")