from params import *
import numpy

def density(particles):
    rho = numpy.zeros(NG)
    for p in range(len(particles)):
        i = particles[p].get_i()
        h = particles[p].pos - (i * dx)

        nxt = (i + 1) if (i + 1 < NG) else 0
        rho[i] += particles[p].q * (dx - h)
        rho[nxt] += particles[p].q * h
    
    # rho[NG-1] = (rho[NG-1] + rho[0]) * 0.5
    # rho[NG-1] += rho[0]
    # rho[0] = rho[NG-1]

    rho /= (dx * dx)

    return rho

def potential(rho):
    rho_k = numpy.fft.fft(rho)
    phi_k = numpy.copy(rho_k)
    i = 0.0 + 1.0j
    W = numpy.exp(2 * i * numpy.pi / NG)
    Wm = 1.0 + 0.0j
    for m in range(NG):
        denom = 2.0 + 0.0j
        denom -= Wm + 1.0/Wm
        if denom != 0:
            phi_k[m] *= dx * dx / denom
        Wm *= W

    phi = numpy.fft.ifft(phi_k)
    phi = numpy.real(phi)

    return phi

def field_n(phi):
    E = numpy.zeros(NG)
    for i in range(NG):
        nxt_i = i + 1 if (i < NG-1) else 0
        pvr_i = i - 1 if (i > 0) else NG-1

        E[i] = (phi[pvr_i] - phi[nxt_i]) / (dx * 2)

    return E

def field_p(field, particles):
    E = numpy.zeros(len(particles))
    for p in range(len(particles)):
        if particles[p].move:
            i = particles[p].get_i()
            h = particles[p].pos - (i * dx)
            nxt = (i + 1) if (i + 1 < NG) else 0

            E[p] += field[i] * (dx - h) + field[nxt] * h

    E /= dx

    return E

def update(particles, field):
    for p in range(len(particles)):
        if particles[p].move:
            particles[p].vel += field[p] * particles[p].qm * dt
            particles[p].pos += particles[p].vel * dt

            particles[p].pos = particles[p].pos % L
    return particles
def outphase(direction, particles, field):
    dT = 0.5 * direction * dt
    for p in range(len(particles)):
        if particles[p].move:
            particles[p].vel += field[p] * particles[p].qm * dT
    return particles



