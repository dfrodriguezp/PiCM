from cycle import *
from params import *
from starter import *
import numpy
from tqdm import tqdm
import os

def main():
    positions, velocities, charges = coldPlasma()
    # positions, velocities, charges = twoStreamUniform()

    folders = ("/phase_space", "/field", "/potential", "/density", "/energy")
    for i in range(len(folders)):
        os.system("mkdir -p results{}".format(folders[i]))

    energy = open("results/energy/energy.dat", "w")
    
    for step in tqdm(range(steps)):
        nodeIndex = numpy.array(positions / dx, dtype=int)
        h = positions - (nodeIndex * dx)

        indexesInNode = [numpy.where(nodeIndex == node)[0] for node in range(NG)]
        indexesMovesInNode = [indexesInNode[node][indexesInNode[node] < NP] for node in range(NG)]

        rho = density(charges, indexesInNode, h)
        phi = potential(rho)
        E_n = field_n(phi)
        E_p = field_p(E_n, indexesMovesInNode, h)

        if step == 0:
            outphase(-1.0, velocities, charges, E_p)

        update(positions, velocities, charges, E_p)

        finalVelocities = numpy.copy(velocities)

        outphase(1.0, finalVelocities, charges, E_p)

        # Write data
        if step % 10 == 0:
            phase_space = open("results/phase_space/step_{}.dat".format(step), "w")
            Efield = open("results/field/step_{}.dat".format(step), "w")
            potentialOutput = open("results/potential/step_{}.dat".format(step), "w")
            densityOutput = open("results/density/step_{}.dat".format(step), "w")

        KE = 0.0
        FE = 0.0

        for i in range(NG):
            FE += rho[i] * phi[i]
            if step % 10 == 0:
                Efield.write("{} {}\n".format(i*dx, E_n[i]))
                potentialOutput.write("{} {}\n".format(i*dx, phi[i]))
                densityOutput.write("{} {}\n".format(i*dx, rho[i]))

        for p in range(NP):
            KE += (charges[p] / numpy.sign(charges[p])) * finalVelocities[p] * finalVelocities[p]
            if step % 10 == 0:
                phase_space.write("{} {}\n".format(positions[p], finalVelocities[p]))

        energy.write("{} {} {}\n".format(KE, FE, step))
        phase_space.close()
        Efield.close()
        densityOutput.close()
        potentialOutput.close()

    energy.close()
    
    # rho_test, phi_test, E_n_test = numpy.loadtxt("test/grid_test.txt", unpack=True)
    # pos_test, vel_test, E_p_test = numpy.loadtxt("test/particles_test.txt", unpack=True)

    # assert(numpy.allclose(pos_test, positions))
    # assert(numpy.allclose(vel_test, velocities))
    # assert(numpy.allclose(rho_test, rho))
    # assert(numpy.allclose(phi_test, phi))
    # assert(numpy.allclose(E_n_test, E_n))
    # assert(numpy.allclose(E_p_test, E_p))

if __name__ == '__main__':
    main()