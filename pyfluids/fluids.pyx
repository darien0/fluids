
cimport pyfluids.fluids as fluids
import numpy as np

def inverse_dict(d):
    return dict((v,k) for k, v in d.iteritems())

_riemannsolvers  = {"hll"    : FLUIDS_RIEMANN_HLL,
                    "hllc"   : FLUIDS_RIEMANN_HLLC,
                    "exact"  : FLUIDS_RIEMANN_EXACT}
_riemannsolvers_i = inverse_dict(_riemannsolvers)

cdef class FluidDescriptor(object):
    """
    Class that describes the microphysics (equation of state) and required
    buffer sizes for a FluidState.
    """
    def __cinit__(self):
        self._c = fluids_descr_new()

    def __dealloc__(self):
        fluids_descr_del(self._c)

    def __init__(self):
        fluids_descr_setfluid(self._c, FLUIDS_NRHYD)
        fluids_descr_setgamma(self._c, 1.4);
        fluids_descr_seteos(self._c, FLUIDS_EOS_GAMMALAW);

    property nprimitive:
        def __get__(self):
            return fluids_descr_getncomp(self._c, FLUIDS_PRIMITIVE)
    property npassive:
        def __get__(self):
            return fluids_descr_getncomp(self._c, FLUIDS_PASSIVE)
    property ngravity:
        def __get__(self):
            return fluids_descr_getncomp(self._c, FLUIDS_GRAVITY)
    property nmagnetic:
        def __get__(self):
            return fluids_descr_getncomp(self._c, FLUIDS_MAGNETIC)
    property nlocation:
        def __get__(self):
            return fluids_descr_getncomp(self._c, FLUIDS_LOCATION)


cdef class FluidState(object):
    """
    Class that holds fluid variables, and caches them for future calls. These
    objects will accumulate lots of memory unless the cache is disabled (which
    it is by default), or the erase_cache() method is used after each time a
    member function is invoked.
    """
    def __cinit__(self):
        self._c = fluids_state_new()

    def __dealloc__(self):
        fluids_state_del(self._c)

    def __init__(self, FluidDescriptor D):
        fluids_state_setdescr(self._c, D._c)
        self._np = D.nprimitive
        self._ns = D.npassive
        self._ng = D.ngravity
        self._nm = D.nmagnetic
        self._nl = D.nlocation
        self._disable_cache = 1
        self._descr = D

    property primitive:
        def __get__(self):
            cdef np.ndarray[np.double_t,ndim=1] x = np.zeros(self._np)
            fluids_state_getattr(self._c, <double*>x.data, FLUIDS_PRIMITIVE)
            return x
        def __set__(self, np.ndarray[np.double_t,ndim=1] x):
            if x.size != self._np: raise ValueError("wrong size input array")
            fluids_state_setattr(self._c, <double*>x.data, FLUIDS_PRIMITIVE)
    property passive:
        def __get__(self):
            cdef np.ndarray[np.double_t,ndim=1] x = np.zeros(self._ns)
            fluids_state_getattr(self._c, <double*>x.data, FLUIDS_PASSIVE)
            return x
        def __set__(self, np.ndarray[np.double_t,ndim=1] x):
            if x.size != self._ns: raise ValueError("wrong size input array")
            fluids_state_setattr(self._c, <double*>x.data, FLUIDS_PASSIVE)
    property gravity:
        def __get__(self):
            cdef np.ndarray[np.double_t,ndim=1] x = np.zeros(self._ng)
            fluids_state_getattr(self._c, <double*>x.data, FLUIDS_GRAVITY)
            return x
        def __set__(self, np.ndarray[np.double_t,ndim=1] x):
            if x.size != self._ng: raise ValueError("wrong size input array")
            fluids_state_setattr(self._c, <double*>x.data, FLUIDS_GRAVITY)
    property magnetic:
        def __get__(self):
            cdef np.ndarray[np.double_t,ndim=1] x = np.zeros(self._nm)
            fluids_state_getattr(self._c, <double*>x.data, FLUIDS_MAGNETIC)
            return x
        def __set__(self, np.ndarray[np.double_t,ndim=1] x):
            if x.size != self._ng: raise ValueError("wrong size input array")
            fluids_state_setattr(self._c, <double*>x.data, FLUIDS_MAGNETIC)
    property location:
        def __get__(self):
            cdef np.ndarray[np.double_t,ndim=1] x = np.zeros(self._nl)
            fluids_state_getattr(self._c, <double*>x.data, FLUIDS_LOCATION)
            return x
        def __set__(self, np.ndarray[np.double_t,ndim=1] x):
            if x.size != self._ng: raise ValueError("wrong size input array")
            fluids_state_setattr(self._c, <double*>x.data, FLUIDS_LOCATION)

    def from_conserved(self, np.ndarray[np.double_t,ndim=1] x):
        if x.size != self._np: raise ValueError("wrong size input array")
        fluids_state_fromcons(self._c, <double*>x.data, FLUIDS_CACHE_RESET)
        if self._disable_cache: fluids_state_erasecache(self._c)

    def conserved(self):
        cdef np.ndarray[np.double_t,ndim=1] x = np.zeros(self._np)
        fluids_state_derive(self._c, <double*>x.data, FLUIDS_CONSERVED)
        if self._disable_cache: fluids_state_erasecache(self._c)
        return x

    def eigenvalues(self, dim=0):
        cdef int flag = [FLUIDS_EVAL0, FLUIDS_EVAL1, FLUIDS_EVAL2][dim]
        cdef np.ndarray[np.double_t,ndim=1] x = np.zeros(self._np)
        fluids_state_derive(self._c, <double*>x.data, flag)
        if self._disable_cache: fluids_state_erasecache(self._c)
        return x

    def left_eigenvectors(self, dim=0):
        cdef int flag = [FLUIDS_LEVECS0, FLUIDS_LEVECS1, FLUIDS_LEVECS2][dim]
        cdef np.ndarray[np.double_t,ndim=2] x = np.zeros([self._np]*2)
        fluids_state_derive(self._c, <double*>x.data, flag)
        if self._disable_cache: fluids_state_erasecache(self._c)
        return x

    def right_eigenvectors(self, dim=0):
        cdef int flag = [FLUIDS_REVECS0, FLUIDS_REVECS1, FLUIDS_REVECS2][dim]
        cdef np.ndarray[np.double_t,ndim=2] x = np.zeros([self._np]*2)
        fluids_state_derive(self._c, <double*>x.data, flag)
        if self._disable_cache: fluids_state_erasecache(self._c)
        return x

    def sound_speed(self):
        cdef double cs2
        fluids_state_derive(self._c, &cs2, FLUIDS_SOUNDSPEEDSQUARED)
        if self._disable_cache: fluids_state_erasecache(self._c)
        return cs2**0.5

    def erase_cache(self):
        fluids_state_erasecache(self._c)

    def enable_cache(self):
        self._disable_cache = 0

    def disable_cache(self):
        self._disable_cache = 1


class FluidStateVector(object):
    """
    Class that holds many fluid states in a Numpy array.
    """
    def __init__(self, shape, descr):
        self._descr = descr
        self._states = np.ndarray(shape=shape, dtype=FluidState)
        self._np = descr.nprimitive
        self._shape = tuple(shape)
        for i in range(self._states.size):
            self._states.flat[i] = FluidState(self._descr)
        self.primitive = np.zeros(self._shape + (self._np,))

    def __getitem__(self, *args):
        return self._states.__getitem__(*args)

    @property
    def primitive(self):
        P = np.zeros([self._states.size, self._np])
        for n, S in enumerate(self._states.flat):
            P[n] = S.primitive
        return P.reshape(self._shape + (self._np,))

    @primitive.setter
    def primitive(self, P):
        if P.shape != self._shape + (self._np,):
            raise ValueError("wrong shape input array")
        Q = P.reshape([self._states.size, self._np])
        for n, S in enumerate(self._states.flat):
            S.primitive = Q[n]

    @property
    def conserved(self):
        U = np.zeros([self._states.size, self._np])
        for n, S in enumerate(self._states.flat):
            U[n] = S.conserved()
        return U.reshape(self._shape + (self._np,))

    @conserved.setter
    def conserved(self, U):
        if U.shape != self._shape + (self._np,):
            raise ValueError("wrong shape input array")
        V = U.reshape([self._states.size, self._np])
        for n, S in enumerate(self._states.flat):
            S.from_conserved(V[n])


cdef class RiemannSolver(object):
    """
    Class which represents a two-state riemann solver.
    """
    def __cinit__(self):
        self._c = fluids_riemn_new()
        self.SL = None
        self.SR = None

    def __dealloc__(self):
        fluids_riemn_del(self._c)

    def __init__(self):
        fluids_riemn_setsolver(self._c, FLUIDS_RIEMANN_EXACT)

    property solver:
        def __get__(self):
            cdef int solver
            fluids_riemn_getsolver(self._c, &solver)
            return _riemannsolvers_i[solver]
        def __set__(self, val):
            fluids_riemn_setsolver(self._c, _riemannsolvers[val])

    def set_states(self, FluidState SL, FluidState SR):
        if SL._descr is not SR._descr:
            raise ValueError("different fluid descriptor on left and right")
        self.SL = SL # hold onto these so they're not deleted
        self.SR = SR
        fluids_riemn_setdim(self._c, 0)
        fluids_riemn_setstateL(self._c, SL._c)
        fluids_riemn_setstateR(self._c, SR._c)
        fluids_riemn_execute(self._c)

    def sample(self, double s):
        if self.SL is None or self.SR is None:
            raise ValueError("solver needs a need a left and right state")
        cdef FluidState S = FluidState(self.SL._descr)
        fluids_riemn_sample(self._c, S._c, s) # sets S.primitive
        return S
