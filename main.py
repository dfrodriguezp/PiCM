import cycle
import numpy
from tqdm import tqdm
import os
import click

@click.command()
@click.option("-sample", default="state.dat")
def main(sample):
    positions = numpy.loadtxt(sample, usecols=(0, 1), unpack=True).T
    velocities = numpy.loadtxt(sample, usecols=(2, 3, 4), unpack=True).T
    charges, move = numpy.loadtxt(sample, usecols=(5, 6), unpack=True)
    QoverM = numpy.sign(charges)

    # folders = ("/phase_space", "/field")
    # for i in range(len(folders)):
    #     os.system("mkdir -p results{}".format(folders[i]))
    NGx = 256
    NGy = 1
    steps = 1000
    Lx = 4 * numpy.pi
    Ly = 1.0
    dx = Lx / NGx
    dy = Ly / NGy
    dt = 0.1
    NP = len(positions)
    move_indexes, = numpy.where(move == 1)

    Bext = numpy.array([[0.0, 0.0, 0.0]] * len(move_indexes))

    # Auxiliary vectors for Boris algorithm
    a = 0.5 * QoverM[move_indexes, numpy.newaxis] * Bext * dt
    a_2 = numpy.linalg.norm(a, axis=1) * numpy.linalg.norm(a, axis=1)
    b = (2 * a) / (1 + a_2[:, numpy.newaxis])

    for step in tqdm(range(steps)):
        
        # in case that initial velocites in y are zero, the following test must pass 
        assert(numpy.allclose(positions[:, 1], numpy.zeros_like(positions[:, 1])))
        
        currentNodesX = numpy.array(positions[:, 0] / dx, dtype=int)
        currentNodesY = numpy.array(positions[:, 1] / dy, dtype=int)

        hx = positions[:, 0] - (currentNodesX * dx)
        hy = positions[:, 1] - (currentNodesY * dy)
        nxtX = (currentNodesX + 1) % NGx
        nxtY = (currentNodesY + 1) % NGy

        rho = cycle.density(NP, NGx, NGy, dx, dy, charges, currentNodesX, currentNodesY, hx, hy, nxtX, nxtY)
        phi = cycle.potential(NGx, NGy, dx, dy, rho)
        E_n = cycle.field_n(NGx, NGy, dx, dy, phi)
        E_p = cycle.field_p(NP, dx, dy, E_n, currentNodesX, currentNodesY, hx, hy, nxtX, nxtY, move_indexes)

        if step == 0:
            cycle.outphase(a, b, -1.0, velocities, QoverM, E_p, Bext, dt, move_indexes)

        cycle.update(a, b, positions, velocities, QoverM, E_p, Bext, dt, Lx, Ly, move_indexes)

        final_velocities = numpy.copy(velocities)

        cycle.outphase(a, b, 1.0, final_velocities, QoverM, E_p, Bext, dt, move_indexes)

        # Write data
        # if step % 10 == 0:
        #     phase_space = open("results/phase_space/step_{}.dat".format(step), "w")
        #     Efield = open("results/field/step_{}.dat".format(step), "w")
        #     for i in range(NG):
        #         Efield.write("{} {}\n".format(i*dx, E_n[i]))
        #     for p in range(NP):
        #         if final_parts[p].move:
        #             phase_space.write("{} {}\n".format(final_parts[p].pos, final_parts[p].vel))
        #     phase_space.close()
        #     Efield.close()

    rho_test, phi_test, E_n_test = numpy.loadtxt("test/grid_test.txt", unpack=True)
    pos_test, vel_test, E_p_test = numpy.loadtxt("test/particles_test.txt", unpack=True)

    # for i in range(NP):
    #     print(pos_test[i], positions[:, 0][i], pos_test[i] == positions[:, 0][i])

    # print(E_p_test.shape, E_p.shape)
    # print(rho.shape)
    # for i in range(NGx):
    #     print(rho_test[i], rho[i][0], rho_test[i] == rho[i][0])
    assert(numpy.allclose(pos_test, positions[:, 0]))
    assert(numpy.allclose(vel_test, velocities[:, 0]))
    assert(numpy.allclose(rho_test, rho[:, 0]))
    assert(numpy.allclose(phi_test, phi[:, 0]))
    assert(numpy.allclose(E_n_test, E_n[:, 0, 0]))
    assert(numpy.allclose(E_p_test, E_p[:, 0]))

if __name__ == '__main__':
    main()