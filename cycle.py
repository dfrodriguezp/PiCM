import numpy

def density(NP, NGx, NGy, dx, dy, charges, currentNodesX, currentNodesY, hx, hy, nxtX, nxtY):
    rho = numpy.zeros(shape=(NGx, NGy))
    for i in range(NP):
        rho[currentNodesX[i], currentNodesY[i]] += charges[i] * (dx - hx[i]) * (dy - hy[i])
        rho[currentNodesX[i], nxtY[i]] += charges[i] * (dx - hx[i]) * hy[i]
        rho[nxtX[i], currentNodesY[i]] += charges[i] * hx[i] * (dy - hy[i])
        rho[nxtX[i], nxtY[i]] += charges[i] * hx[i] * hy[i]

    rho /= (dx * dy * dx * dy)

    return rho

def potential(NGx, NGy, dx, dy, rho):
    rho_k = numpy.fft.fftn(rho)
    Wx = numpy.exp(2 * 1j * numpy.pi / NGx)
    Wy = numpy.exp(2 * 1j * numpy.pi / NGy)
    Wn = 1.0 + 0.0j
    Wm = 1.0 + 0.0j
    dx_2 = dx * dx
    dy_2 = dy * dy

    for n in range(NGx):
        for m in range(NGy):
            denom = dy_2 * (2 - Wn - 1.0/Wn) + dx_2 * (2 - Wm - 1.0/Wm)
            if denom != 0:
                rho_k[n, m] *= dx_2 * dy_2 / denom
            Wm *= Wy
        Wn *= Wx

    phi = numpy.fft.ifftn(rho_k)
    phi = numpy.real(phi)

    return phi

def field_n(NGx, NGy, dx, dy, phi):
    E = numpy.zeros(shape=(2, NGx, NGy))
    for j in range(NGy):
        for i in range(NGx):
            nxt_i = (i + 1) % NGx
            prv_i = (i - 1) % NGx

            E[0, i, j] = (phi[prv_i, j] - phi[nxt_i, j]) / (dx * 2)

    for i in range(NGx):
        for j in range(NGy):
            nxt_j = (j + 1) % NGy
            prv_j = (j - 1) % NGy

            E[1, i, j] = (phi[i, prv_j] - phi[i, nxt_j]) / (dy * 2)

    return E

def field_p(NP, dx, dy, E_n, currentNodesX, currentNodesY, hx, hy, nxtX, nxtY, move_indexes):
    E = numpy.zeros(shape=(2, NP))

    for i in move_indexes:
        A = (dx - hx[i]) * (dy - hy[i])
        B = (dx - hx[i]) * hy[i]
        C = hx[i] * (dy - hy[i])
        D = hx[i] * hy[i]
        
        E[:, i] += E_n[:, currentNodesX[i], currentNodesY[i]] * A \
                 + E_n[:, currentNodesX[i], nxtY[i]] * B \
                 + E_n[:, nxtX[i], currentNodesY[i]] * C \
                 + E_n[:, nxtX[i], nxtY[i]] * D

    E /= (dx * dy)

    return E

def update(positions, velocities, charges, E_p, dt, Lx, Ly):
    velocities[:, 0] += E_p[0, :] * numpy.sign(charges) * dt
    velocities[:, 1] += E_p[1, :] * numpy.sign(charges) * dt
    positions += velocities * dt

    # positions[:, 0] = numpy.fmod(positions[:, 0], Lx)
    # positions[:, 1] = numpy.fmod(positions[:, 1], Ly)

    positions[:, 0] %= Lx
    positions[:, 1] %= Ly

    assert(numpy.all(positions[:, 0] < Lx))
    assert(numpy.all(positions[:, 1] < Ly))

def outphase(direction, velocities, charges, E_p, dt):
    dT = 0.5 * direction * dt
    velocities[:, 0] += E_p[0, :] * numpy.sign(charges) * dT
    velocities[:, 1] += E_p[1, :] * numpy.sign(charges) * dT