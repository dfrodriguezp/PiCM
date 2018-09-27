from params import *
import numpy

def density(charges, nodeIndex, h, nxt):
    rho = numpy.zeros(NG)
    for i, node in enumerate(nodeIndex):
        rho[node] += charges[i] * (dx - h[i])
        rho[nxt[i]] += charges[i] * h[i]

    rho /= (dx * dx)

    return rho

def potential(rho):
    rho_k = numpy.fft.fft(rho)
    i = 0.0 + 1.0j
    W = numpy.exp(2 * i * numpy.pi / NG)
    Wm = 1.0 + 0.0j

    for m in range(NG):
        denom = 2.0 + 0.0j
        denom -= Wm + 1.0/Wm
        if denom != 0:
            rho_k[m] *= dx * dx / denom
        Wm *= W

    phi = numpy.fft.ifft(rho_k)
    phi = numpy.real(phi)

    return phi

def field_n(phi):
    E = numpy.zeros(NG)
    for i in range(NG):
        nxt = (i + 1) % NG
        prv = (i - 1) % NG

        E[i] = (phi[prv] - phi[nxt]) / (dx * 2)

    return E

def field_p(field, nodeIndex, h, nxt):
    E = numpy.zeros(NP * 2)

    for i in range(NP):
        E[i] += field[nodeIndex[i]] * (dx - h[i]) + field[nxt[i]] * h[i]

    E /= dx

    return E

def update(positions, velocities, charges, field):
    velocities += field * numpy.sign(charges) * dt
    positions += velocities * dt
    positions = positions % L

def outphase(direction, velocities, charges, field):
    dT = 0.5 * direction * dt
    velocities += field * numpy.sign(charges) * dT