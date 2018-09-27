from cycle import *
from params import *
from starter import *
import numpy
from tqdm import tqdm
import copy
import os
from matplotlib import pyplot

def main():
    particles = cold_plasma()
    # particles = twoStream1()

    folders = ("/phase_space", "/field")
    for i in range(len(folders)):
        os.system("mkdir -p results{}".format(folders[i]))

    for step in tqdm(range(steps)):
        rho = density(particles)
        phi = potential(rho)
        E_n = field_n(phi)
        E_p = field_p(E_n, particles)

        if step == 0:
            particles = outphase(-1.0, particles, E_p)

        particles = update(particles, E_p)

        final_parts = copy.deepcopy(particles)

        final_parts = outphase(1.0, final_parts, E_p)

        # # Write data
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

    os.system("mkdir -p test/")
    
    grid_test = open("test/grid_test.txt", mode="w")
    for i, _ in enumerate(rho):
        grid_test.write("{} {} {}\n".format(rho[i], phi[i], E_n[i]))
    grid_test.close()

    particles_test = open("test/particles_test.txt", mode="w")
    for i, _ in enumerate(particles):
        particles_test.write("{} {} {}\n".format(particles[i].pos, particles[i].vel, E_p[i]))
    particles_test.close()

if __name__ == '__main__':
    main()