from params import *
import numpy

def coldPlasma():
    margin = dx / 5
    mobilePos = numpy.linspace(margin, L - margin, NP)
    fixedPos = numpy.linspace(margin, L - margin, NP)

    ks = 2 * n * numpy.pi / L

    mobilePos = (mobilePos + A * numpy.cos(ks * mobilePos)) % L
    velocities = numpy.zeros(NP * 2)
    charges = (L / (NP * 2)) * numpy.concatenate((-numpy.ones(NP), numpy.ones(NP)))
    positions = numpy.concatenate((mobilePos, fixedPos))

    return numpy.array(positions, dtype=float), numpy.array(velocities, dtype=float), numpy.array(charges, dtype=float)


def twoStreamUniform():
    sep = L / (NP / 2)
    ks = 2 * n * numpy.pi / L
    positions = list()

    for i in range(int(NP / 2)):
        # unperturbed position
        x0 = (i + 0.5) * sep

        # perturbation
        dX = A * numpy.cos(ks * x0)
        x1 = x0 + dX
        x2 = x0 - dX

        # periodic boundaries
        x1 = x1 % L
        x2 = x2 % L

        positions.append(x1)
        positions.append(x2)

    sep = L / NP
    for i in range(NP):
        x0 = (i + 0.5) * sep
        positions.append(x0)

    velocities = numpy.concatenate((numpy.ones(NP), numpy.zeros(NP)))
    velocities[1::2] *= -1

    charges = (L / (NP * 2)) * numpy.concatenate((-numpy.ones(NP), numpy.ones(NP)))

    return numpy.array(positions, dtype=float), numpy.array(velocities, dtype=float), numpy.array(charges, dtype=float)


def twoStreamRandom():
    margin = dx / 5
    mobilePos = numpy.random.uniform(margin, L - margin, size=NP)
    fixedPos = numpy.random.uniform(margin, L - margin, size=NP)

    velocities = numpy.concatenate((numpy.random.choice([-1.0, 1.0], size=NP), numpy.zeros(NP)))
    charges = (L / NP) * numpy.concatenate((-numpy.ones(NP), numpy.ones(NP)))
    positions = numpy.concatenate((mobilePos, fixedPos))

    return numpy.array(positions, dtype=float), numpy.array(velocities, dtype=float), numpy.array(charges, dtype=float)
