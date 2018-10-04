from cycle import *
from starter import *
import numpy
from tqdm import tqdm
import os
import click

@click.command()
@click.option("-sample", default="state.dat")
def main(sample):
    positions = numpy.loadtxt(sample, usecols=(0, 1), unpack=True).T
    velocities = numpy.loadtxt(sample, usecols=(2, 3), unpack=True).T
    charges, move = numpy.loadtxt(sample, usecols=(4, 5), unpack=True)

    # folders = ("/phase_space", "/field")
    # for i in range(len(folders)):
    #     os.system("mkdir -p results{}".format(folders[i]))
    NGx = 256
    NGy = 1
    steps = 1000
    Lx = 4 * numpy.pi
    Ly = 4 * numpy.pi
    dx = Lx / NGx
    dy = Ly / NGy
    dt = 0.1
    n = 1


    NP = len(positions)
    move_indexes, = numpy.where(move == 1)
    for step in tqdm(range(steps)):
        
        # in case that initial velocites in y are zero, the following test must pass 
        assert(numpy.allclose(positions[:, 1], numpy.zeros_like(positions[:, 1])))
        
        currentNodesX = numpy.array(positions[:, 0] / dx, dtype=int)
        currentNodesY = numpy.array(positions[:, 1] / dy, dtype=int)


        hx = positions[:, 0] - (currentNodesX * dx)
        hy = positions[:, 1] - (currentNodesY * dy)
        nxtX = (currentNodesX + 1) % NGx
        nxtY = (currentNodesY + 1) % NGy

        rho = density(NP, NGx, NGy, dx, dy, charges, currentNodesX, currentNodesY, hx, hy, nxtX, nxtY)
        phi = potential(NGx, NGy, dx, dy, rho)
        E_n = field_n(NGx, NGy, dx, dy, phi)
        E_p = field_p(NP, dx, dy, E_n, currentNodesX, currentNodesY, hx, hy, nxtX, nxtY, move_indexes)

        if step == 0:
            outphase(-1.0, velocities, charges, E_p, dt)

        update(positions, velocities, charges, E_p, dt, Lx, Ly)

        # final_velocities = numpy.copy(velocities)

        # outphase(1.0, final_velocities, charges, E_p, dt)

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

    for i in range(NGx):
        print(phi_test[i], phi[i][0], phi_test[i] == phi[i][0])
    # assert(numpy.allclose(pos_test, positions[:, 0]))
    # assert(numpy.allclose(vel_test, velocities[:, 0]))
    # assert(numpy.allclose(rho_test, rho))
    assert(numpy.allclose(phi_test, phi))
    # assert(numpy.allclose(E_n_test, E_n))
    # assert(numpy.allclose(E_p_test, E_p))

if __name__ == '__main__':
    main()