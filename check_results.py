import numpy
import json
import click

@click.command()
@click.argument("jsonfile")
def main(jsonfile):
    with open(jsonfile, "r") as json_input:
        root = json.load(json_input)

    steps = root["steps"]
    ss_freq = root["ss_frequency"]

    for i in range(steps):
        if i % ss_freq == 0:
            c_x, c_y, c_vx, c_vy, c_vz = numpy.loadtxt("cpp_test_results/phase_space/step_{}_.dat".format(i), unpack=True)
            p_x, p_y, p_vx, p_vy, p_vz = numpy.loadtxt("python_test_results/phase_space/step_{}_.dat".format(i), unpack=True)
            c_Ex, c_Ey = numpy.loadtxt("cpp_test_results/Efield/step_{}_.dat".format(i), usecols=(2, 3), unpack=True)
            p_Ex, p_Ey = numpy.loadtxt("python_test_results/Efield/step_{}_.dat".format(i), usecols=(2, 3), unpack=True)
            c_phi = numpy.loadtxt("cpp_test_results/phi/step_{}_.dat".format(i), usecols=(2,), unpack=True)
            p_phi = numpy.loadtxt("python_test_results/phi/step_{}_.dat".format(i), usecols=(2,), unpack=True)
            c_rho = numpy.loadtxt("cpp_test_results/rho/step_{}_.dat".format(i), usecols=(2,), unpack=True)
            p_rho = numpy.loadtxt("python_test_results/rho/step_{}_.dat".format(i), usecols=(2,), unpack=True)


            assert(numpy.allclose(c_Ex, p_Ex)) 
            assert(numpy.allclose(c_Ey, p_Ey)) 
            assert(numpy.allclose(c_phi, p_phi)) 
            assert(numpy.allclose(c_rho, p_rho)) 
            assert(numpy.allclose(c_x, p_x)) 
            assert(numpy.allclose(c_y, p_y)) 
            assert(numpy.allclose(c_vx, p_vx)) 
            assert(numpy.allclose(c_vy, p_vy)) 
            assert(numpy.allclose(c_vz, p_vz))

    c_KE, c_FE = numpy.loadtxt("cpp_test_results/energy/energy.dat", usecols=(1, 2), unpack=True) 
    p_KE, p_FE = numpy.loadtxt("python_test_results/energy/energy.dat", usecols=(1, 2), unpack=True) 
    
    assert(numpy.allclose(c_KE, p_KE))
    assert(numpy.allclose(c_FE, p_FE))

    print("All match!") 

if __name__ == '__main__':
    main()