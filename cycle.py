import numpy

def density(NP, NG, dx, charges, current, h, nxt):
    rho = numpy.zeros(shape=(NG, NG))
    for p in range(NP):
        rho[current[0, p], current[1, p]] += charges[p] * (dx - h[0, p]) * (dx - h[1, p])
        rho[current[0, p], nxt[1, p]] += charges[p] * (dx - h[0, p]) * h[1, p]
        rho[nxt[0, p], current[1, p]] += charges[p] * h[0, p] * (dx - h[1, p])
        rho[nxt[0, p], nxt[1, p]] += charges[p] * h[0, p] * h[1, p]

    rho /= (dx * dx * dx * dx)

    return rho

def potential(NG, dx, rho):
    rho_k = numpy.fft.fftn(rho)
    i = 0.0 + 1.0j
    W = numpy.exp(2 * i * numpy.pi / NG)
    Wn = 1.0 + 0.0j
    Wm = 1.0 + 0.0j

    for n in range(NG):
        for m in range(NG):
            denom = 4.0 + 0.0j
            denom -= Wn + 1.0/Wn + Wm + 1.0/Wm
            if denom != 0:
                rho_k[n, m] *= dx * dx / denom
            Wm *= W
        Wn *= W

    phi = numpy.fft.ifftn(rho_k)
    phi = numpy.real(phi)

    return phi

def field_n(NG, dx, phi):
    E = numpy.zeros(shape=(2, NG, NG))
    for j in range(NG):
        for i in range(NG):
            nxt_i = (i + 1) % NG
            prv_i = (i - 1) % NG

            E[0, i, j] = (phi[prv_i, j] - phi[nxt_i, j])

    for i in range(NG):
        for j in range(NG):
            nxt_j = (j + 1) % NG
            prv_j = (j - 1) % NG

            E[1, i, j] = (phi[i, prv_j] - phi[i, nxt_j])

    E /= (dx * 2)

    return E

def field_p(NP, dx, field, current, h, nxt, move_index):
    E = numpy.zeros(shape=(2, NP))

    for i in move_index:
        A = (dx - h[0, i]) * (dx - h[1, i])
        B = (dx - h[0, i]) * h[1, i]
        C = h[0, i] * (dx - h[1, i])
        D = h[0, i] * h[1, i]
 
        E[:, i] += field[:, current[0, i], current[1, i]] * A \
                 + field[:, current[0, i], nxt[1, i]] * B \
                 + field[:, nxt[0, i], current[1, i]] * C \
                 + field[:, nxt[0, i], nxt[1, i]] * D

    E /= (dx * dx)

    return E

def update(positions, velocities, charges, field, dt, L):
    velocities += field * numpy.sign(charges) * dt
    positions += velocities * dt
    positions %= L

def outphase(direction, velocities, charges, field, dt):
    dT = 0.5 * direction * dt
    velocities += field * numpy.sign(charges) * dT