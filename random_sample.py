import numpy
import json
import os

N_electrons = 1000
N_ions = 1000
N = N_electrons + N_ions

Lx = 10
Ly = 10

output = "example"
os.system("mkdir -p {}".format(output))
filename = "{}/sample.dat".format(output)
samplefile = open(filename, "w")
samplefile.write("# x y vx vy vz QoverM move\n")

for i in range(N_electrons):
    x = numpy.random.uniform(0, Lx)
    y = numpy.random.uniform(0, Ly)
    vx = numpy.random.uniform(-0.5, 0.5)
    vy = numpy.random.uniform(-0.5, 0.5)
    vz = numpy.random.uniform(-0.5, 0.5)
    samplefile.write("{} {} {} {} {} {} {}\n".format(
        x, y, vx, vy, vz, -1.0, 1))

for i in range(N_ions):
    x = numpy.random.uniform(0, Lx)
    y = numpy.random.uniform(0, Ly)
    vx = numpy.random.uniform(-0.5, 0.5)
    vy = numpy.random.uniform(-0.5, 0.5)
    vz = numpy.random.uniform(-0.5, 0.5)
    samplefile.write("{} {} {} {} {} {} {}\n".format(
        x, y, vx, vy, vz, 1.0, 1))

samplefile.close()

json_data = {
    "N": N,
    "steps": 1000,
    "grid_size": [
        64,
        64
    ],
    "sys_length": [
        Lx,
        Ly
    ],
    "dt": 0.1,
    "output": output,
    "sample": filename,
    "results": [
        "electric_potential",
        "electric_field",
        "velocities"
    ]
}

with open("{}/example.json".format(output), "w") as data:
    json.dump(json_data, data)
