from params import *
import numpy

def cold_plasma():
    margin = dx / 5
    mobilePos = numpy.linspace(margin, L - margin, NP)
    fixedPos = numpy.linspace(margin, L - margin, NP)

    ks = 2 * n * numpy.pi / L

    mobilePos = (mobilePos + A * numpy.cos(ks * mobilePos)) % L
    velocities = numpy.zeros(NP * 2)
    charges = (L / NP) * numpy.concatenate((-numpy.ones(NP), numpy.ones(NP)))
    positions = numpy.concatenate((mobilePos, fixedPos))

    return numpy.array(positions, dtype=float), numpy.array(velocities, dtype=float), numpy.array(charges, dtype=float)

def twoStream1():
    parts = []
    sep = L / (NP / 2)
    ks = 2 * n * numpy.pi / L

    for i in range(int(NP / 2)):
        # unperturbed position
        x0 = (i + 0.5) * sep
        # perturbation
        theta = ks * x0
        dX = A * numpy.cos(theta)
        x1 = x0 + dX
        x2 = x0 - dX

        # periodic boundaries
        x1 = x1 % L
        x2 = x2 % L

        # add to parts
        parts.append(Particle(x1, 1.0, -1.0, True))
        parts.append(Particle(x2, -1.0, -1.0, True))

    sep = L / NP
    for i in range (NP):
        x0 = (i + 0.5) * sep
        parts.append(Particle(x0, 0.0, 1.0, False))

    return parts
