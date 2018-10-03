from cycle import *
from starter import *
import numpy
from tqdm import tqdm
import os
import click

@click.command()
@click.option("-sample", default="state.dat")
def main(sample):
    positions = numpy.loadtxt(sample, usecols=(0, 1), unpack=True)
    velocities = numpy.loadtxt(sample, usecols=(2, 3), unpack=True)
    charges, move = numpy.loadtxt(sample, usecols=(4, 5), unpack=True)

    # folders = ("/phase_space", "/field")
    # for i in range(len(folders)):
    #     os.system("mkdir -p results{}".format(folders[i]))
    NG = 256
    steps = 2
    L = 4 * numpy.pi
    dx = L / NG
    dt = 0.1
    A = 1e-3
    n = 1

    NP = len(positions[0])
    move_indexes, = numpy.where(move == 1)
    for step in tqdm(range(steps)):
        current = numpy.array(positions / dx, dtype=int)
        h = positions - (current * dx)
        nxt = (current + 1) % NG

        rho = density(NP, NG, dx, charges, current, h, nxt)
        phi = potential(NG, dx, rho)
        E_n = field_n(NG, dx, phi)
        E_p = field_p(NP, dx, E_n, current, h, nxt, move_indexes)

        if step == 0:
            outphase(-1.0, velocities, charges, E_p, dt)

        update(positions, velocities, charges, E_p, dt, L)

        final_velocities = numpy.copy(velocities)

        outphase(1.0, final_velocities, charges, E_p, dt)

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

    # rho_test, phi_test, E_n_test = numpy.loadtxt("test/grid_test.txt", unpack=True)
    # pos_test, vel_test, E_p_test = numpy.loadtxt("test/particles_test.txt", unpack=True)

    # assert(numpy.allclose(pos_test, positions[0]))
    # assert(numpy.allclose(vel_test, velocities[0]))
    # assert(numpy.allclose(rho_test, rho))
    # assert(numpy.allclose(phi_test, phi))
    # assert(numpy.allclose(E_n_test, E_n))
    # assert(numpy.allclose(E_p_test, E_p))

if __name__ == '__main__':
    main()