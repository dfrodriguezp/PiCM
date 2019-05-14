import numpy
import json
import click
from matplotlib import pyplot
import matplotlib
matplotlib.style.use('classic')


@click.command()
@click.argument("jsonfile")
@click.argument("step", default=0)
def main(jsonfile, step):
    with open(jsonfile, "r") as json_input:
        root = json.load(json_input)

    if step >= root["steps"]:
        print("ERROR! You want to plot a snapshot {} but the limit is {}.".format(
            step, root["steps"]-1))
        return

    output = root["output"]

    # Step 0 data for colors
    _, _, vx0, vy0, _ = numpy.loadtxt(
        "{}/phase_space/step_0_.dat".format(output), unpack=True)

    colorsx = numpy.full(len(vx0), "r")
    colorsx[vx0 < 0] = "b"

    colorsy = numpy.full(len(vy0), "r")
    colorsy[vy0 < 0] = "b"

    try:
        x, y, vx, vy, _ = numpy.loadtxt(
            "{}/phase_space/step_{}_.dat".format(output, step), unpack=True)
    except OSError as error:
        print("ERROR! You want to plot a snapshot that doesn't exist.")
        return

    if len(vx) > 1e5:
        colorsx = colorsx[::100]
        colorsy = colorsy[::100]
        x = x[::100]
        y = y[::100]
        vx = vx[::100]
        vy = vy[::100]

    Lx, Ly = root["sys_length"]
    # x - dimension
    pyplot.figure()
    pyplot.title("Step {}".format(step))
    pyplot.scatter(x, vx, c=colorsx)
    pyplot.xlabel("$x$", fontsize=25)
    pyplot.ylabel("$v_x$", fontsize=25)
    pyplot.xlim(0, Lx)
    pyplot.grid()
    pyplot.savefig("{}/phase_space/step_{}_x_.pdf".format(output, step))

    # y - dimension
    pyplot.figure()
    pyplot.title("Step {}".format(step))
    pyplot.scatter(y, vy, c=colorsy)
    pyplot.xlabel("$y$", fontsize=25)
    pyplot.ylabel("$v_y$", fontsize=25)
    pyplot.xlim(0, Ly)
    pyplot.grid()
    pyplot.savefig("{}/phase_space/step_{}_y_.pdf".format(output, step))


if __name__ == "__main__":
    main()
