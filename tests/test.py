
import numpy as np
import pyfluids

state = pyfluids.FluidState(fluid='nrhyd')
state.primitive = np.ones(5)

print state.primitive
U = state.conserved()
state.from_conserved(U)

print state.primitive
L = state.left_eigenvectors()
R = state.right_eigenvectors()

print np.dot(L,R)
print state.eigenvalues(dim=1)
print state.sound_speed()


fluid = pyfluids.FluidStateVector([10,10], fluid='nrhyd')
print fluid.descriptor.eos

print fluid.primitive.shape
print fluid.states.shape
print fluid.states[0,0].primitive

fluid.primitive[...] = np.zeros([10,10,5])
