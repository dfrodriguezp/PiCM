import numpy


def density(positions, charges, dx, dy, Nx, Ny, N):
    rho = numpy.zeros(shape=(Nx, Ny))

    for p in range(N):
        i = int(positions[p][0] / dx)
        j = int(positions[p][1] / dy)
        hx = positions[p][0] - (i * dx)
        hy = positions[p][1] - (i * dy)
        nxt_i = (i + 1) % Nx
        nxt_j = (j + 1) % Ny

        rho[i, j] += charges[p] * (dx - hx) * (dy - hy)
        rho[i, nxt_j] += charges[p] * (dx - hx) * hy
        rho[nxt_i, j] += charges[p] * hx * (dy - hy)
        rho[nxt_i, nxt_j] += charges[p] * hx * hy

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


def fieldNodes(phi, dx, dy, Nx, Ny):
    E = numpy.zeros(shape=(Nx, Ny, 3))
    for j in range(Ny):
        for i in range(Nx):
            nxt_i = (i + 1) % Nx
            prv_i = (i - 1) % Nx

            E[i, j, 0] = (phi[prv_i, j] - phi[nxt_i, j]) / (dx * 2)

    for i in range(Nx):
        for j in range(Ny):
            nxt_j = (j + 1) % Ny
            prv_j = (j - 1) % Ny

            E[i, j, 1] = (phi[i, prv_j] - phi[i, nxt_j]) / (dy * 2)

    return E


def fieldParticles(field, positions, moves, dx, dy, Nx, Ny, N):
    E = numpy.zeros(shape=(N, 3))

    for p in range(N):
        if moves[p]:
            i = int(positions[p][0] / dx)
            j = int(positions[p][1] / dy)
            hx = positions[p][0] - (i * dx)
            hy = positions[p][1] - (i * dy)
            nxt_i = (i + 1) % Nx
            nxt_j = (j + 1) % Ny

            A = (dx - hx) * (dy - hy)
            B = (dx - hx) * hy
            C = hx * (dy - hy)
            D = hx * hy

            E[p][0] = field[i][j][0] * A + field[i][nxt_j][0] * B + field[nxt_i][j][0] * C + field[nxt_i][nxt_j][0] * D;
            E[p][1] = field[i][j][1] * A + field[i][nxt_j][1] * B + field[nxt_i][j][1] * C + field[nxt_i][nxt_j][1] * D;

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

