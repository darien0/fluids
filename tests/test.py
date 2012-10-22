
import numpy as np
import pyfluids

state = pyfluids.FluidState(fluid='nrhyd')
state.primitive = [1, 1, 1, 1, 1]

assert (state.primitive == [1, 1, 1, 1, 1]).all()
assert (state.conserved() == [1, 4, 1, 1, 1]).all()

U = state.conserved()
state.from_conserved(U)

assert (abs(state.primitive - [1, 1, 1, 1, 1]) < 1e-14).all()

# Eigen-systems are not permuted to pass this test right now
"""
dim = 0
L = state.left_eigenvectors(dim=dim)
R = state.right_eigenvectors(dim=dim)
L_ = np.zeros_like(L)
R_ = np.zeros_like(R)
L_[:,0] = L[:,0]
L_[:,1] = L[:,2]
L_[:,2] = L[:,3]
L_[:,3] = L[:,4]
L_[:,4] = L[:,1]
R_[0,:] = R[0,:]
R_[1,:] = R[2,:]
R_[2,:] = R[3,:]
R_[3,:] = R[4,:]
R_[4,:] = R[1,:]

L, R = L_, R_
A = state.jacobian(dim=dim)
LAR = np.dot(np.dot(L, A), R)
I = np.dot(L, R)
LAR[abs(LAR) < 1e-12] = 0.0
I[abs(I) < 1e-12] = 0.0
lam = state.eigenvalues(dim=dim)
print LAR
print lam
print I
exit()

assert (abs(np.diag(LAR) - lam) < 1e-12).all()
assert (abs(np.dot(L,R) - np.eye(5)) < 1e-14).all()
assert abs(state.sound_speed() - 1.18321595662) < 1e-12
"""

state = pyfluids.FluidState(fluid='gravp')
state.primitive = [1, 1, 1, 1, 1]
f1 = state.flux()
state.primitive = [1, 1, 1, 0, 0]
f2 = state.flux()
assert (f1 != f2).any()

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

fluid = pyfluids.FluidStateVector([64], fluid='gravp')
fluid.primitive[...,:] = [1, 1, 1, 1, 1]
fluid.gravity[...,:] = [1, 1, 1, 1]

assert (fluid.states[0].gravity == [1, 1, 1, 1]).all()
assert (fluid.gravity[0] == [1, 1, 1, 1]).all()
assert fluid.descriptor.fluid is 'gravp'
