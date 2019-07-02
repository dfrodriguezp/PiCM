import numpy


def density(NGx, NGy, dx, dy, hx, hy, currentNodesX, currentNodesY, nxtX, nxtY, indexesInNode, charges):
    '''Computes particle-to-grid interpolation and returns the values of the charge density on each grid point.

    Parameters:
        NGx, NGy (int): number of grid points in the x and y dimensions.
        dx, dy (float): cell size.
        hx, hy (array): particles relative coordinates on each cell.
        currentNodesX, currentNodesY (array): origin node for each cell.
        nxtX, nxtY (array): neighbor nodes of each origin node.
        indexesinNode (array): particles indexes that are in each cell.
        charges (array): electric charge of each particle.

    Returns:
        rho (2D-array): charge density in each node.

    '''
    rho = numpy.zeros(shape=(NGx, NGy))

    for node in range(NGx * NGy):
        i = indexesInNode[node] # Get the particles indexes located in the current cell

        # Particle-to-grid interpolation. Add the charge contributions in each node
        rho[currentNodesX[i], currentNodesY[i]
            ] += numpy.sum(charges[i] * (dx - hx[i]) * (dy - hy[i]))
        rho[currentNodesX[i], nxtY[i]
            ] += numpy.sum(charges[i] * (dx - hx[i]) * hy[i])
        rho[nxtX[i], currentNodesY[i]
            ] += numpy.sum(charges[i] * hx[i] * (dy - hy[i]))
        rho[nxtX[i], nxtY[i]] += numpy.sum(charges[i] * hx[i] * hy[i])

    # Divide for cell size squared to obtain correct dimensions
    rho /= (dx * dy * dx * dy)

    return rho


def potential(NGx, NGy, dx, dy, rho):
    '''Returns the values of the electric potential in each grid point

    Parameters:
        NGx, NGy (int): number of grid points in the x and y dimensions.
        dx, dy (float): cell size.
        rho (2D-array): charge density
    
    Returns:
        phi (2D-array): electric potential value in each node.
    '''
    
    # Discrete Fourier transform (DFT) of rho
    rho_k = numpy.fft.fftn(rho)

    Wx = numpy.exp(2 * 1j * numpy.pi / NGx)
    Wy = numpy.exp(2 * 1j * numpy.pi / NGy)
    Wm = 1.0 + 0.0j
    Wn = 1.0 + 0.0j
    dx_2 = dx * dx
    dy_2 = dy * dy

    # Find the transform of phi in terms of the transform of rho
    for m in range(NGx):
        for n in range(NGy):
            denom = dy_2 * (2 - Wm - 1.0/Wm) + dx_2 * (2 - Wn - 1.0/Wn)
            if denom != 0:
                rho_k[m, n] *= dx_2 * dy_2 / denom
            Wn *= Wy
        Wm *= Wx

    # Compute phi with the real part of the inverse DFT 
    phi = numpy.fft.ifftn(rho_k)
    phi = numpy.real(phi)

    return phi


def field_n(NGx, NGy, dx, dy, phi):
    '''Returns the values of the electric field in each grid point

    Parameters:
        NGx, NGy (int): number of grid points in the x and y dimensions.
        dx, dy (float): cell size.
        phi (2D-array): electric potential
    
    Returns:
        E (3D-array): electric field components in each node.
    '''
    E = numpy.zeros(shape=(NGx, NGy, 3))

    # Compute centered finite difference on x component
    for j in range(NGy):
        for i in range(NGx):
            nxt_i = (i + 1) % NGx # Modulo operator for PBC
            prv_i = (i - 1) % NGx

            E[i, j, 0] = (phi[prv_i, j] - phi[nxt_i, j]) / (dx * 2)

    # Compute centered finite difference on y component
    for i in range(NGx):
        for j in range(NGy):
            nxt_j = (j + 1) % NGy
            prv_j = (j - 1) % NGy

            E[i, j, 1] = (phi[i, prv_j] - phi[i, nxt_j]) / (dy * 2)

    return E


def field_p(NP, dx, dy, E_n, currentNodesX, currentNodesY, hx, hy, nxtX, nxtY, move_indexes):
    '''Computes backwards interpolation and returns the value of the electric field on each particle.

    Parameters:
        NP (int): number of particles.
        dx, dy (float): cell size.
        E_n (3D-array): electric field components in each node.
        currentNodesX, currentNodesY (array): origin node for each cell.
        hx, hy (array): particles relative coordinates on each cell.
        nxtX, nxtY (array): neighbor nodes of each origin node.
        move_indexes (array): indexes of moving particles.
    
    Returns:
        E (2D-array): electric field components on each particle.
    '''
    E = numpy.zeros(shape=(NP, 3))

    # Weight factors
    A = (dx - hx[move_indexes]) * (dy - hy[move_indexes])
    B = (dx - hx[move_indexes]) * hy[move_indexes]
    C = hx[move_indexes] * (dy - hy[move_indexes])
    D = hx[move_indexes] * hy[move_indexes]

    # Backwards interpolation. Adds the field contribution from the surrounding nodes into each particle
    E[move_indexes, :] += E_n[currentNodesX[move_indexes], currentNodesY[move_indexes], :] * A[:, numpy.newaxis] \
        + E_n[currentNodesX[move_indexes], nxtY[move_indexes], :] * B[:, numpy.newaxis] \
        + E_n[nxtX[move_indexes], currentNodesY[move_indexes], :] * C[:, numpy.newaxis] \
        + E_n[nxtX[move_indexes], nxtY[move_indexes],
              :] * D[:, numpy.newaxis]

    # Divide by cell size to obtain correct dimensions
    E /= (dx * dy)

    return E


def boris(v1, v2, velocities, QoverM, E_p, dt, move_indexes):
    '''Updates velocities using the Boris method.

    Parameters:
        v1, v2 (array): auxiliary vectors of the Boris method.
        velocities (array): velocity vector of each particle.
        QoverM (array): q / m ratio of ech particle.
        E_p (2D-array): electric field acting on each particle.
        dt (float): time step.
        move_indexes (array): indexes of moving particles.
    
    Returns:
        None
    '''

    # Step 1
    v_minus = velocities[move_indexes] + 0.5 * \
        QoverM[move_indexes, numpy.newaxis] * E_p[move_indexes] * dt

    # Step 2
    v_prime = v_minus + numpy.cross(v_minus, v1)

    # Step 3
    v_plus = v_minus + numpy.cross(v_prime, v2)

    # Step 4
    velocities[move_indexes] = v_plus + 0.5 * \
        QoverM[move_indexes, numpy.newaxis] * E_p[move_indexes] * dt


def update(v1, v2, positions, velocities, QoverM, E_p, dt, Lx, Ly, move_indexes):
    '''Updates particles position and velocity

    Parameters:
        v1, v2 (array): auxiliary vectors of the Boris method.
        velocities (array): velocity vector of each particle.
        QoverM (array): q / m ratio of ech particle.
        E_p (2D-array): electric field acting on each particle.
        dt (float): time step.
        Lx, Ly (float): length of the system.
        move_indexes (array): indexes of moving particles.
    
    Returns:
        None
    '''

    # Update velocities with Boris method
    boris(v1, v2, velocities, QoverM, E_p, dt, move_indexes)

    # Update position and apply periodic boundary condition
    positions += velocities[:, (0, 1)] * dt
    positions[:, 0] %= Lx
    positions[:, 1] %= Ly

    # Assert positions are within boundaries
    assert(numpy.all(positions[:, 0] < Lx))
    assert(numpy.all(positions[:, 1] < Ly))


def outphase(v1, v2, direction, velocities, QoverM, E_p, dt, move_indexes):
    '''Performs leapgfrog scheme on velocities

    Parameters:
        v1, v2 (array): auxiliary vectors of the Boris method.
        direction (int): forward (+1) or backward (-1) velocity.
        velocities (array): velocity vector of each particle.
        QoverM (array): q / m ratio of ech particle.
        E_p (2D-array): electric field acting on each particle.
        dt (float): time step.
        move_indexes (array): indexes of moving particles.
    
    Returns:
        None
    '''

    # New time outphased by half dt
    dT = 0.5 * direction * dt

    # Update velocities with Boris method
    boris(v1, v2, velocities, QoverM, E_p, dT, move_indexes)
