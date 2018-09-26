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
            outphase(-1.0, particles, E_p)

        update(particles, E_p)

        final_parts = copy.deepcopy(particles)

        outphase(1.0, final_parts, E_p)

        # Write data
        if step % 10 == 0:
            phase_space = open("results/phase_space/step_{}.dat".format(step), "w")
            Efield = open("results/field/step_{}.dat".format(step), "w")
            for i in range(NG):
                Efield.write("{} {}\n".format(i*dx, E_n[i]))
            for p in range(NP):
                if final_parts[p].move:
                    phase_space.write("{} {}\n".format(final_parts[p].pos, final_parts[p].vel))
            phase_space.close()
            Efield.close()


    positions, velocities = numpy.loadtxt("test/test.txt", unpack=True)
    
    pos = [p.pos for p in particles]
    assert(numpy.allclose(pos, positions))

    vel = [p.vel for p in particles]
    assert(numpy.allclose(vel, velocities))


if __name__ == '__main__':
    main()