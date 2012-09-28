
import cProfile
import pstats
import pyfluids
import numpy as np

def set_primitive(fluid):
    P = np.ones(fluid.shape + (5,))
    for n in range(10):
        fluid.set_primitive(P)

def set_conserved(fluid):
    U = np.ones(fluid.shape + (5,))
    U[:,1] = 4.0
    for n in range(10):
        fluid.set_conserved(U)

def main():
    descr = pyfluids.FluidDescriptor()
    fluid = pyfluids.FluidStateVector([32,32,32], descr)
    for S in fluid.flat:
        S.enable_cache()

    set_primitive(fluid)
    set_conserved(fluid)

cProfile.run('main()', 'perf.pstats')
p = pstats.Stats('perf.pstats')
p.sort_stats('time').print_stats()
