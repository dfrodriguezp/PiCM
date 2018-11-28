import numpy


def density(positions, charges, dx, dy, Nx, Ny, N):
    rho = numpy.zeros(shape=(NGx, NGy))

    for p in range(N):
        i = int(positions[p][0] / dx)
        j = int(positions[p][1] / dy)
        hx = positions[p][0] - (i * dx)
        hy = positions[p][1] - (i * dy)
        nxt_i = (i + 1) % Nx
        nxt_j = (j + 1) % Ny

        rho[i][j] += charges[p] * (dx - hx) * (dy - hy)
        rho[i][nxt_j] += charges[p] * (dx - hx) * hy
        rho[nxt_i][j] += charges[p] * hx * (dy - hy)
        rho[nxt_i][nxt_j] += charges[p] * hx * hy

    rho /= (dx * dy * dx * dy)

    return rho


def potential(rho, dx, dy, Nx, Ny):
    rho_k = numpy.fft.fftn(rho)
    Wx = numpy.exp(2 * 1j * numpy.pi / Nx)
    Wy = numpy.exp(2 * 1j * numpy.pi / Ny)
    Wn = 1.0 + 0.0j
    Wm = 1.0 + 0.0j
    dx_2 = dx * dx
    dy_2 = dy * dy

    for n in range(Nx):
        for m in range(Ny):
            denom = dy_2 * (2 - Wn - 1.0/Wn) + dx_2 * (2 - Wm - 1.0/Wm)
            if denom != 0:
                rho_k[n, m] *= dx_2 * dy_2 / denom
            Wm *= Wy
        Wn *= Wx

    phi = numpy.fft.ifftn(rho_k)
    phi = numpy.real(phi)

    return phi


def field_n(NGx, NGy, dx, dy, phi):
    E = numpy.zeros(shape=(NGx, NGy, 3))
    for j in range(NGy):
        for i in range(NGx):
            nxt_i = (i + 1) % NGx
            prv_i = (i - 1) % NGx

            E[i, j, 0] = (phi[prv_i, j] - phi[nxt_i, j]) / (dx * 2)

    for i in range(NGx):
        for j in range(NGy):
            nxt_j = (j + 1) % NGy
            prv_j = (j - 1) % NGy

            E[i, j, 1] = (phi[i, prv_j] - phi[i, nxt_j]) / (dy * 2)

    return E


def field_p(NP, dx, dy, E_n, currentNodesX, currentNodesY, hx, hy, nxtX, nxtY, move_indexes):
    E = numpy.zeros(shape=(NP, 3))

    A = (dx - hx[move_indexes]) * (dy - hy[move_indexes])
    B = (dx - hx[move_indexes]) * hy[move_indexes]
    C = hx[move_indexes] * (dy - hy[move_indexes])
    D = hx[move_indexes] * hy[move_indexes]

    E[move_indexes, :] += E_n[currentNodesX[move_indexes], currentNodesY[move_indexes], :] * A[:, numpy.newaxis] \
                        + E_n[currentNodesX[move_indexes], nxtY[move_indexes], :] * B[:, numpy.newaxis] \
                        + E_n[nxtX[move_indexes], currentNodesY[move_indexes], :] * C[:, numpy.newaxis] \
                        + E_n[nxtX[move_indexes], nxtY[move_indexes], :] * D[:, numpy.newaxis]

    E /= (dx * dy)

    return E


def boris(v1, v2, velocities, QoverM, E_p, Bext, dt, move_indexes):
    v_minus = velocities[move_indexes] + 0.5 * QoverM[move_indexes, numpy.newaxis] * E_p[move_indexes] * dt
    v_prime = v_minus + numpy.cross(v_minus, v1)
    v_plus = v_minus + numpy.cross(v_prime, v2)
    velocities[move_indexes] = v_plus + 0.5 * QoverM[move_indexes, numpy.newaxis] * E_p[move_indexes] * dt


def update(v1, v2, positions, velocities, QoverM, E_p, Bext, dt, Lx, Ly, move_indexes):
    boris(v1, v2, velocities, QoverM, E_p, Bext, dt, move_indexes)
    positions += velocities[:, (0, 1)] * dt

    positions[:, 0] %= Lx
    positions[:, 1] %= Ly

    assert(numpy.all(positions[:, 0] < Lx))
    assert(numpy.all(positions[:, 1] < Ly))


def outphase(v1, v2, direction, velocities, QoverM, E_p, Bext, dt, move_indexes):
    dT = 0.5 * direction * dt
    boris(v1, v2, velocities, QoverM, E_p, Bext, dT, move_indexes)

