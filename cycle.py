import numpy

def density(NG, dx, charges, current, h, nxt):
    rho = numpy.zeros(NG, NG)
    for i, node in enumerate(current):
        rho[node] += charges[i] * (dx - h[i])
        rho[nxt[i]] += charges[i] * h[i]

    rho /= (dx * dx)

    return rho

def potential(NG, dx, rho):
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

def field_n(NG, dx, phi):
    E = numpy.zeros(NG)
    for i in range(NG):
        nxt = (i + 1) % NG
        prv = (i - 1) % NG

        E[i] = (phi[prv] - phi[nxt]) / (dx * 2)

    return E

def field_p(dx, field, nodeIndex, h, nxt, move_index):
    E = numpy.zeros(len(nodeIndex))

    for i in move_index:
        E[i] += field[nodeIndex[i]] * (dx - h[i]) + field[nxt[i]] * h[i]

    E /= dx

    return E

def update(positions, velocities, charges, field, dt, L):
    velocities += field * numpy.sign(charges) * dt
    positions += velocities * dt
    positions %= L

def outphase(direction, velocities, charges, field, dt):
    dT = 0.5 * direction * dt
    velocities += field * numpy.sign(charges) * dT