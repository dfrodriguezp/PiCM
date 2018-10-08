import numpy

NP = 2048
NG = 256
steps = 1000
L = 4 * numpy.pi
dx = L / NG
dt = 0.1
A = 1e-3
n = 1

def cold_plasma():
    margin = dx / 5
    mobilePos = numpy.linspace(margin, L - margin, NP)
    fixedPos = numpy.linspace(margin, L - margin, NP)

    ks = 2 * n * numpy.pi / L

    mobilePos = (mobilePos + A * numpy.cos(ks * mobilePos)) % L
    velocities = numpy.zeros(NP * 2)
    charges = (L / NP) * numpy.concatenate((-numpy.ones(NP), numpy.ones(NP)))
    move = numpy.concatenate((numpy.ones(NP), numpy.zeros(NP)))
    positions = numpy.concatenate((mobilePos, fixedPos))

    return positions, numpy.zeros_like(positions), velocities, numpy.zeros_like(velocities), numpy.zeros_like(velocities), charges, move

def main():
    px, py, vx, vy, vz, charges, move = cold_plasma()
    output = open("state.dat", "w")
    for i, _ in enumerate(px):
        output.write("{} {} {} {} {} {} {}\n".format(px[i], py[i], vx[i], vy[i], vz[i], charges[i], move[i]))
    output.close()

if __name__ == '__main__':
    main()