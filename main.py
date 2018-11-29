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
        print("ERROR! JSON file not found.")
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
    dx = root.get("dx", 1.0)
    dy = root.get("dy", 1.0)


    N = root.get("N", None)
    if not N:
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
    
    Nx, Ny = gridSize 
    Lx = dx * Nx
    Ly = dy * Ny

    positions = numpy.loadtxt(samplefile, usecols=(0, 1), unpack=True).T
    velocities = numpy.loadtxt(samplefile, usecols=(2, 3, 4), unpack=True).T
    QoverM, moves = numpy.loadtxt(samplefile, usecols=(5, 6), unpack=True)
    charges = Lx * Ly * QoverM / N
    masses = charges / QoverM

    move_indexes, = numpy.where(moves == 1)

    folders = ["/energy"]
    if writePhaseSpace: folders.append("/phase_space")
    if writeEfield: folders.append("/Efield")
    if writePhi: folders.append("/phi")
    if writeRho: folders.append("/rho")

    for f in folders:
        os.system("mkdir -p {}".format(outputName + f))

    energy = open("{}/energy/energy_seed_{}_.dat".format(outputName, seed), "w")

    print("Simulations running...\n")
    for step in tqdm(range(steps)):
        writeStep = (step % ss_freq == 0)
        RHO = cycle.density(positions, charges, dx, dy, Nx, Ny, N)
        PHI = cycle.potential(RHO, dx, dy, Nx, Ny)
        EFIELDn = cycle.fieldNodes(PHI, dx, dy, Nx, Ny)
        EFIELDp = cycle.fieldParticles(EFIELDn, positions, move_indexes, dx, dy, Nx, Ny, N)

        if step == 0:
            cycle.outphase(-1.0, velocities, QoverM, move_indexes, EFIELDp, Bext, dt, N)

        cycle.update(positions, velocities, QoverM, move_indexes, EFIELDp, Bext, Lx, Ly, dt, N)
        new_velocities = numpy.copy(velocities)
        cycle.outphase(1.0, new_velocities, QoverM, move_indexes, EFIELDp, Bext, dt, N)

        if (writePhaseSpace and writeStep):
            phaseSpace = open("{}/phase_space/step_{}_seed_{}_.dat".format(outputName, step, seed), "w")
            phaseSpace.write("# x y vx vy vz\n\n")
        if (writeEfield and writeStep):
            electricField = open("{}/Efield/step_{}_seed_{}_.dat".format(outputName, step, seed), "w")
            electricField.write("# x y Ex Ey\n\n")
        if (writePhi and writeStep):
            electricPotential = open("{}/phi/step_{}_seed_{}_.dat".format(outputName, step, seed), "w")
            electricPotential.write("# x y phi\n\n")
        if (writeRho and writeStep):
            chargeDensity = open("{}/rho/step_{}_seed_{}_.dat".format(outputName, step, seed), "w")
            chargeDensity.write("# x y rho\n\n")

        KE = 0.0
        FE = 0.0

        for p in move_indexes:
            if (writePhaseSpace and writeStep):
                phaseSpace.write("{} {} {} {} {}\n".format(*positions[p], *new_velocities[p]))

            KE += masses[p] * numpy.linalg.norm(new_velocities[p]) * numpy.linalg.norm(new_velocities[p])

        KE *= 0.5

        for i in range(Nx):
            for j in range(Ny):
                if (writeEfield and writeStep):
                    electricField.write("{} {} {} {}\n".format(i * dx, j * dy, EFIELDn[i][j][0], EFIELDn[i][j][1]))
                if (writePhi and writeStep):
                    electricPotential.write("{} {} {}\n".format(i * dx, j * dy, PHI[i][j]))
                if (writeRho and writeStep):
                    chargeDensity.write("{} {} {}\n".format(i * dx, j * dy, RHO[i][j]))

                FE += RHO[i][j] * PHI[i][j]

        FE *= 0.5

        energy.write("{} {} {}\n".format(step, KE, FE))
        
        if (writePhaseSpace and writeStep): phaseSpace.close()
        if (writeEfield and writeStep): electricField.close()
        if (writePhi and writeStep): electricPotential.close()
        if (writeRho and writeStep): chargeDensity.close()

    energy.close()

    print("\nSimulation finished!\n")


if __name__ == '__main__':
    main()