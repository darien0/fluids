
import numpy as np
import pyfluids

state = pyfluids.FluidState(fluid='nrhyd')
state.primitive = [1, 1, 1, 1, 1]

assert (state.primitive == [1, 1, 1, 1, 1]).all()
assert (state.conserved() == [1, 4, 1, 1, 1]).all()

U = state.conserved()
state.from_conserved(U)

assert (abs(state.primitive - [1, 1, 1, 1, 1]) < 1e-14).all()

L = state.left_eigenvectors()
R = state.right_eigenvectors()

assert (abs(np.dot(L,R) - np.eye(5)) < 1e-14).all()
assert abs(state.sound_speed() - 1.18321595662) < 1e-12

fluid = pyfluids.FluidStateVector([10,10], fluid='nrhyd')
assert fluid.descriptor.eos is 'gammalaw'

fluid.states[0,0].primitive = [1.0, 2.0, 1.0, 1.0, 1.0]
assert (fluid.primitive[0,0] == [1.0, 2.0, 1.0, 1.0, 1.0]).all()

fluid.primitive[0,0] = 3.0
assert (fluid.states[0,0].primitive == 3.0).all()

fluid.primitive = 1.0
assert (abs(fluid.sound_speed() - 1.18321595662) < 1e-12).all()

fluid.primitive = 2.0
U = fluid.conserved()
assert U.shape == (10,10,5)

fluid.from_conserved(U)
assert (abs(fluid.primitive - 2.0) < 1e-14).all()

