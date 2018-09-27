from cycle import *
from params import *
from starter import *
import numpy
from tqdm import tqdm
import copy
import os
from matplotlib import pyplot

def main():
    positions, velocities, charges, moves = cold_plasma()
    # particles = twoStream1()

    folders = ("/phase_space", "/field")
    for i in range(len(folders)):
        os.system("mkdir -p results{}".format(folders[i]))

    for step in range(steps):
        nodeIndex = numpy.array([int(positions[i] / dx) for i in range(len(positions))])
        h = positions - (nodeIndex * dx)
        nxt = (nodeIndex + 1) % NG

        rho = density(charges, nodeIndex, h, nxt)
        phi = potential(rho)
        E_n = field_n(phi)
        E_p = field_p(E_n, positions, velocities, moves, nodeIndex, h, nxt)

        if step == 0:
            outphase(-1.0, velocities, charges, moves, E_p)

        update(positions, velocities, charges, moves, E_p)

        # final_velocities = copy.deepcopy(velocities)

        # outphase(1.0, E_p, final_velocities, charges, moves)
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


    rho_test, phi_test, E_n_test, E_p_test = numpy.loadtxt("test/grid_test.txt", unpack=True)
    pos_test, vel_test = numpy.loadtxt("test/particles_test.txt", unpack=True)

    # for i, _ in enumerate(positions):
    #     print(pos_test[i], positions[i], pos_test[i] == positions[i])
    # assert(numpy.allclose(pos_test, positions))
    # assert(numpy.allclose(vel_test, velocities))
    # assert(numpy.allclose(rho_test, rho))
    # assert(numpy.allclose(phi_test, phi))
    # assert(numpy.allclose(E_n_test, E_n))
    # assert(numpy.allclose(E_p_test, E_p))


if __name__ == '__main__':
    main()