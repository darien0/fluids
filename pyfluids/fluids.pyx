
cimport pyfluids.fluids as fluids
import numpy as np

def inverse_dict(d):
    return dict((v,k) for k, v in d.iteritems())


_fluidsystem = {"nrhyd"         : FLUIDS_NRHYD,
                "gravs"         : FLUIDS_GRAVS}
_coordsystem = {"cartesian"     : FLUIDS_COORD_CARTESIAN,
                "spherical"     : FLUIDS_COORD_SPHERICAL,
                "cylindrical"   : FLUIDS_COORD_CYLINDRICAL}
_equationofstate = {"gammalaw"  : FLUIDS_EOS_GAMMALAW,
                    "tabulated" : FLUIDS_EOS_TABULATED}
_riemannsolver = {"hll"         : FLUIDS_RIEMANN_HLL,
                  "hllc"        : FLUIDS_RIEMANN_HLLC,
                  "exact"       : FLUIDS_RIEMANN_EXACT}

_fluidsystem_i     = inverse_dict(_fluidsystem)
_coordsystem_i     = inverse_dict(_coordsystem)
_equationofstate_i = inverse_dict(_equationofstate)
_riemannsolver_i   = inverse_dict(_riemannsolver)


cdef class FluidDescriptor(object):
    """
    Class that describes the microphysics (equation of state) and required
    buffer sizes for a FluidState.
    """
    def __cinit__(self):
        self._c = fluids_descr_new()

    def __dealloc__(self):
        fluids_descr_del(self._c)

    def __init__(self, fluid='nrhyd', coordsystem='cartesian', eos='gammalaw',
                 gamma=1.4):
        fluids_descr_setfluid(self._c, _fluidsystem[fluid])
        fluids_descr_seteos(self._c, _equationofstate[eos])
        fluids_descr_setcoordsystem(self._c, _coordsystem[coordsystem])
        fluids_descr_setgamma(self._c, gamma)

    property fluid:
        def __get__(self):
            cdef int val
            fluids_descr_getfluid(self._c, &val)
            return _fluidsystem_i[val]

    property eos:
        def __get__(self):
            cdef int val
            fluids_descr_geteos(self._c, &val)
            return _equationofstate_i[val]

    property coordsystem:
        def __get__(self):
            cdef int val
            fluids_descr_getcoordsystem(self._c, &val)
            return _coordsystem_i[val]

    property gammalawindex:
        def __get__(self):
            cdef double val
            fluids_descr_getgamma(self._c, &val)
            return val

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

    def __init__(self, *args, **kwargs):
        cdef FluidDescriptor D
        try:
            D = args[0]
        except:
            D = FluidDescriptor(**kwargs)
        self._descr = D
        fluids_state_setdescr(self._c, D._c)
        self._np = D.nprimitive
        self._ns = D.npassive
        self._ng = D.ngravity
        self._nm = D.nmagnetic
        self._nl = D.nlocation

    property descriptor:
        def __get__(self):
            return self._descr
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
        fluids_state_fromcons(self._c, <double*>x.data, FLUIDS_CACHE_DEFAULT)

    def conserved(self):
        cdef np.ndarray[np.double_t,ndim=1] x = np.zeros(self._np)
        fluids_state_derive(self._c, <double*>x.data, FLUIDS_CONSERVED)
        return x

    def eigenvalues(self, dim=0):
        cdef int flag = [FLUIDS_EVAL0, FLUIDS_EVAL1, FLUIDS_EVAL2][dim]
        cdef np.ndarray[np.double_t,ndim=1] x = np.zeros(self._np)
        fluids_state_derive(self._c, <double*>x.data, flag)
        return x

    def left_eigenvectors(self, dim=0):
        cdef int flag = [FLUIDS_LEVECS0, FLUIDS_LEVECS1, FLUIDS_LEVECS2][dim]
        cdef np.ndarray[np.double_t,ndim=2] x = np.zeros([self._np]*2)
        fluids_state_derive(self._c, <double*>x.data, flag)
        return x

    def right_eigenvectors(self, dim=0):
        cdef int flag = [FLUIDS_REVECS0, FLUIDS_REVECS1, FLUIDS_REVECS2][dim]
        cdef np.ndarray[np.double_t,ndim=2] x = np.zeros([self._np]*2)
        fluids_state_derive(self._c, <double*>x.data, flag)       
        return x

    def sound_speed(self):
        cdef double cs2
        fluids_state_derive(self._c, &cs2, FLUIDS_SOUNDSPEEDSQUARED)
        return cs2**0.5


cdef class FluidStateVector(FluidState):
    def __init__(self, shape, *args, **kwargs):
        super(FluidStateVector, self).__init__(*args, **kwargs)
        shape = tuple(shape)
        self._states = np.ndarray(shape=shape, dtype=FluidState)
        self._primitive = np.zeros(shape + (self._np,))
        self._passive = np.zeros(shape + (self._ns,))
        self._gravity = np.zeros(shape + (self._ng,))
        self._magnetic = np.zeros(shape + (self._nm,))
        self._location = np.zeros(shape + (self._nl,))
        for i in range(self._states.size):
            self._states.flat[i] = FluidState(self.descriptor)

    property states:
        def __get__(self):
            return self._states
    property primitive:
        def __get__(self):
            return self._primitive
    property passive:
        def __get__(self):
            return self._passive
    property gravity:
        def __get__(self):
            return self._gravity
    property magnetic:
        def __get__(self):
            return self._magnetic
    property location:
        def __get__(self):
            return self._location


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
            return _riemannsolver_i[solver]
        def __set__(self, val):
            fluids_riemn_setsolver(self._c, _riemannsolver[val])

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
