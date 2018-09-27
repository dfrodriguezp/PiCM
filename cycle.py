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
        nxt_i = i + 1 if ((i+1) < NG) else 0
        pvr_i = i - 1 if (i > 0) else NG-1

        E[i] = (phi[pvr_i] - phi[nxt_i]) / (dx * 2)

    return E

def field_p(field, positions, velocities, charges, moves, nodeIndex, h, nxt):
    E = numpy.zeros(len(positions))

    for i, _ in enumerate(positions):
        if moves[i]:
            E[i] += field[nodeIndex[i]] * (dx - h[i]) + field[nxt[i]] * h[i]

    E /= dx

    return E

def update(positions, velocities, charges, moves, field):
    for i in range(len(positions)):
        if moves[i]:
            velocities[i] += field[i] * numpy.sign(charges[i]) * dt
            positions[i] += velocities[i] * dt

            positions[i] = positions[i] % L
    
def outphase(direction, field, velocities, charges, moves):
    dT = 0.5 * direction * dt
    for i, _ in enumerate(velocities):
        if moves[i]:
            velocities[i] += field[i] * numpy.sign(charges[i]) * dT