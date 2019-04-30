import cycle
import numpy
from tqdm import tqdm
import os
import click
import json
import sys


@click.command()
@click.argument("jsonfile")
def main(jsonfile):
    if not os.path.isfile(jsonfile):
        print("ERROR! Please include a parameters JSON file.")
        sys.exit()

    with open(jsonfile, "r") as json_input:
        try:
            root = json.load(json_input)
        except json.decoder.JSONDecodeError as error:
            print("ERROR! Corrupted JSON file.\n")
            print(error)
            sys.exit()

    results = root.get("results", [])
    samplefile = root.get("sample", None)
    outputName = root.get("output", "results")

    if not samplefile:
        print("ERROR! Sample file not found in JSON file.")
        sys.exit()

    if not os.path.isfile(samplefile):
        print("ERROR! Sample file not found.")
        sys.exit()

    writePhaseSpace = ("phase_space" in results)
    writeEfield = ("electric_field" in results)
    writePhi = ("electric_potential" in results)
    writeRho = ("charge_density" in results)

    steps = root.get("steps", 50)
    ss_freq = root.get("ss_frequency", 10)
    seed = root.get("seed", 69696969)
    dt = root.get("dt", 0.1)
    # dx = root.get("dx", 1.0)
    # dy = root.get("dy", 1.0)
    Lx = root.get("Lx", 1.0)
    Ly = root.get("Ly", 1.0)

    NP = root.get("N", None)
    if not NP:
        print("ERROR! Number of particles \"N\" not found in JSON file.")
        sys.exit()

    Bext = numpy.array(root.get("Bfield", [0.0, 0.0, 0.0]))
    if len(Bext) != 3:
        print("ERROR! Magnetic field must have three components.")
        sys.exit()

    gridSize = root.get("grid_size", [16, 16])
    if len(gridSize) != 2:
        print("ERROR! Grid size must have two components!")
        sys.exit()

    NGx, NGy = gridSize
    # Lx = dx * NGx
    # Ly = dy * NGy
    dx = Lx / NGx
    dy = Ly / NGy

    sample = root["sample"]
    positions = numpy.loadtxt(sample, usecols=(0, 1), unpack=True).T
    velocities = numpy.loadtxt(sample, usecols=(2, 3, 4), unpack=True).T
    QoverM, moves = numpy.loadtxt(sample, usecols=(5, 6), unpack=True)
    charges = Lx * Ly * QoverM / NP
    masses = charges / QoverM

    move_indexes, = numpy.where(moves == 1)
    Bext = numpy.full((len(move_indexes), 3), Bext)

    folders = ["/energy"]
    if writePhaseSpace:
        folders.append("/phase_space")
    if writeEfield:
        folders.append("/Efield")
    if writePhi:
        folders.append("/phi")
    if writeRho:
        folders.append("/rho")

    for f in folders:
        os.system("mkdir -p {}".format(outputName + f))

    # Auxiliary vectors for Boris algorithm v1 and v2

    # v1 is usually named t and v2 is usually named s, but I don't want t
    # to be confused with time.

    v1 = 0.5 * QoverM[move_indexes, numpy.newaxis] * Bext * dt
    # Magnitude of v1 squared
    v1_2 = numpy.linalg.norm(v1, axis=1) * numpy.linalg.norm(v1, axis=1)
    v2 = (2 * v1) / (1 + v1_2[:, numpy.newaxis])

    energy = open(
        "{}/energy/energy_seed_{}_.dat".format(outputName, seed), "w")

    print("Simulation running...\n")
    for step in tqdm(range(steps)):
        writeStep = (step % ss_freq == 0)
        currentNodesX = numpy.array(positions[:, 0] / dx, dtype=int)
        currentNodesY = numpy.array(positions[:, 1] / dy, dtype=int)
        currentX_currentY = currentNodesX + currentNodesY * NGx

        hx = positions[:, 0] - (currentNodesX * dx)
        hy = positions[:, 1] - (currentNodesY * dy)
        nxtX = (currentNodesX + 1) % NGx
        nxtY = (currentNodesY + 1) % NGy

        indexesInNode = numpy.array([numpy.where(currentX_currentY == node)[
                                    0] for node in range(NGx * NGy)])

        rho = cycle.density(NGx, NGy, dx, dy, hx, hy, currentNodesX,
                            currentNodesY, nxtX, nxtY, indexesInNode, charges)
        phi = cycle.potential(NGx, NGy, dx, dy, rho)
        E_n = cycle.field_n(NGx, NGy, dx, dy, phi)
        E_p = cycle.field_p(NP, dx, dy, E_n, currentNodesX,
                            currentNodesY, hx, hy, nxtX, nxtY, move_indexes)

        if step == 0:
            cycle.outphase(v1, v2, -1.0, velocities, QoverM,
                           E_p, Bext, dt, move_indexes)

        cycle.update(v1, v2, positions, velocities, QoverM,
                     E_p, Bext, dt, Lx, Ly, move_indexes)

        final_velocities = numpy.copy(velocities)

        cycle.outphase(v1, v2, 1.0, final_velocities,
                       QoverM, E_p, Bext, dt, move_indexes)

        if (writePhaseSpace and writeStep):
            phaseSpace = open(
                "{}/phase_space/step_{}_seed_{}_.dat".format(outputName, step, seed), "w")
            phaseSpace.write("# x y vx vy vz\n")
        if (writeEfield and writeStep):
            electricField = open(
                "{}/Efield/step_{}_seed_{}_.dat".format(outputName, step, seed), "w")
            electricField.write("# x y Ex Ey\n")
        if (writePhi and writeStep):
            electricPotential = open(
                "{}/phi/step_{}_seed_{}_.dat".format(outputName, step, seed), "w")
            electricPotential.write("# x y phi\n")
        if (writeRho and writeStep):
            chargeDensity = open(
                "{}/rho/step_{}_seed_{}_.dat".format(outputName, step, seed), "w")
            chargeDensity.write("# x y rho\n")

        KE = 0.0
        FE = 0.0

        for p in move_indexes:
            if (writePhaseSpace and writeStep):
                phaseSpace.write("{} {} {} {} {}\n".format(
                    *positions[p], *final_velocities[p]))

            KE += masses[p] * numpy.linalg.norm(
                final_velocities[p]) * numpy.linalg.norm(final_velocities[p])

        KE *= 0.5

        for i in range(NGx):
            for j in range(NGy):
                if (writeEfield and writeStep):
                    electricField.write("{} {} {} {}\n".format(
                        i * dx, j * dy, E_n[i][j][0], E_n[i][j][1]))
                if (writePhi and writeStep):
                    electricPotential.write(
                        "{} {} {}\n".format(i * dx, j * dy, phi[i][j]))
                if (writeRho and writeStep):
                    chargeDensity.write("{} {} {} {}\n".format(
                        i * dx, j * dy, rho[i][j]))

                FE += rho[i][j] * phi[i][j]

        FE *= 0.5

        energy.write("{} {} {}\n".format(step, KE, FE))

        if (writePhaseSpace and writeStep):
            phaseSpace.close()
        if (writeEfield and writeStep):
            electricField.close()
        if (writePhi and writeStep):
            electricPotential.close()
        if (writeRho and writeStep):
            chargeDensity.close()

    energy.close()

    print("\nSimulation finished!\n")


if __name__ == '__main__':
    main()
