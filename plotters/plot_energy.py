import numpy
import json
import click
from matplotlib import pyplot


@click.command()
@click.argument("jsonfile")
def main(jsonfile):
    with open(jsonfile, "r") as json_input:
        root = json.load(json_input)

    output = root["output"]
    step, KE, FE = numpy.loadtxt(
        "{}/energy/energy.dat".format(output), unpack=True)

    total = KE + FE

    pyplot.figure()
    pyplot.plot(step, KE, linewidth=0.7, label="Kinetic")
    pyplot.plot(step, FE, linewidth=0.7, label="Field")
    pyplot.plot(step, total, linewidth=0.7, label="Total")
    pyplot.xlabel("Steps", fontsize=13)
    pyplot.ylabel("Energy", fontsize=13)
    pyplot.grid(ls="--")
    pyplot.legend(loc=0)
    pyplot.savefig("{}/energy/energy.pdf".format(output))
    pyplot.close()


if __name__ == "__main__":
    main()
