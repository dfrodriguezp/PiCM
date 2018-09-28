from params import *
import numpy


def density(charges, indexes_in_node, h):
    rho = numpy.zeros(NG)   
    for node in range(NG):
        rho[node] += numpy.sum(charges[indexes_in_node[node]] * (dx - h[indexes_in_node[node]]))
        rho[(node + 1) % NG] += numpy.sum(charges[indexes_in_node[node]] * h[indexes_in_node[node]])

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


def field_p(field_nodes, indexes_moves_in_node, h):
    E = numpy.zeros(NP * 2)
    for node in range(NG):
        E[indexes_moves_in_node[node]] += field_nodes[node] * (dx - h[indexes_moves_in_node[node]]) + field_nodes[(node + 1) % NG] * h[indexes_moves_in_node[node]]
    E /= dx
    return E


def update(positions, velocities, charges, field_particles):
    velocities += field_particles * numpy.sign(charges) * dt
    positions += velocities * dt
    positions %= L


def outphase(direction, velocities, charges, field_particles):
    dT = 0.5 * direction * dt
    velocities += field_particles * numpy.sign(charges) * dT