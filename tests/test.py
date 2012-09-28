
import numpy as np
import pyfluids

descr = pyfluids.FluidDescriptor()
state = pyfluids.FluidState(descr)

state.primitive = np.ones(5)

print state.primitive
U = state.conserved()
state.from_conserved(U)

print state.primitive
state.erase_cache()
L = state.left_eigenvectors()
R = state.right_eigenvectors()

print np.dot(L,R)
print state.eigenvalues(dim=1)
print state.sound_speed()

fluid = pyfluids.FluidStateVector([10,10], descr)
print fluid.primitive.shape

fluid.primitive = np.zeros([10,10,5])
fluid.conserved = np.zeros([10,10,5])

R = pyfluids.RiemannSolver()
R.solver = 'hllc'
print R.solver
