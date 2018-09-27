from params import *
import numpy
from matplotlib import pyplot

class Particle(object):
    def __init__(self, pos, vel, qm, move):
        self.pos = pos
        self.vel = vel
        self.qm = qm
        self.q = (1 / self.qm) * (L / NP)
        self.move = move

    def get_i(self):
        return int(self.pos / dx)

def cold_plasma():
    margin = dx / 5
    e_pos = numpy.linspace(margin, L - margin, NP)
    i_pos = numpy.linspace(margin, L - margin, NP)

    ks = 2 * n * numpy.pi / L

    electronPos = (e_pos + A * numpy.cos(ks * e_pos)) % L
    velocities = [0] * 2*NP
    charges = [(1 / -1) * (L / NP)] * NP + [(1 / 1) * (L / NP)] * NP
    moves = [True] * NP + [False] * NP
    positions = list(electronPos) + list(i_pos)
    
    return numpy.array(positions), numpy.array(velocities), numpy.array(charges), numpy.array(moves)

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
