from params import *
import numpy
from matplotlib import pyplot

class Particle(object):
    def __init__(self, pos, vel, qm, move):
        self.pos = pos
        self.vel = vel
        self.qm = qm
        self.q = (1 / self.qm) * (L / (NP * 2))
        self.move = move

    def get_i(self):
        return int(self.pos / dx)

def cold_plasma():
    parts = list()
    margin = dx / 5
    e_pos = numpy.linspace(margin, L - margin, NP)
    i_pos = numpy.linspace(margin, L - margin, NP)

    for i in range(NP):
        x0 = e_pos[i]
        theta = ks * x0
        x1 = x0 + A * numpy.cos(theta)

        x1 = x1 % L

        parts.append(Particle(x1, 0.0, -1.0, True))

    for p in range(NP):
        parts.append(Particle(i_pos[p], 0.0, 1.0, False))

    return parts

def twoStream1():
    parts = []
    sep = L / (NP / 2)
    for i in range(int(NP / 2)):
        # unperturbed position
        x0 = (i + 0.5) * sep

        # perturbation
        theta = ks * x0
        dx = A * numpy.cos(theta)
        x1 = x0 + dx
        x2 = x0 - dx

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



