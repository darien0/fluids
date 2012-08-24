
cimport fluids
import numpy as np

cdef class FluidState(object):
    def __cinit__(self):
        self._c = fluids_new()

    def __dealloc__(self):
        fluids_del(self._c)

    def __init__(self):
        fluids_setfluid(self._c, FLUIDS_NRHYD)
        fluids_alloc(self._c, FLUIDS_FLAGSALL)

    cdef _getattrib(self, double *val, int flag):
        cdef int err = fluids_getattrib(self._c, val, flag)
        if err == FLUIDS_ERROR_INCOMPLETE:
            raise RuntimeError("incomplete state: primitive or conserved "
                               "must already be set")
        elif err != 0:
            raise RuntimeError("unknown error: " + str(err))

    cpdef _map_bufferp(self, np.ndarray buf, int absindex):
        fluids_dealloc(self._c, FLUIDS_PRIMITIVE)
        fluids_mapbuffer(self._c, FLUIDS_PRIMITIVE,
                         <double*>buf.data + absindex * 5)

    cpdef _map_bufferc(self, np.ndarray buf, int absindex):
        fluids_dealloc(self._c, FLUIDS_CONSERVED)
        fluids_mapbuffer(self._c, FLUIDS_CONSERVED,
                         <double*>buf.data + absindex * 5)

    def _set_onlyprimvalid(self):
        fluids_setcacheinvalid(self._c, FLUIDS_FLAGSALL)
        fluids_setcachevalid(self._c, FLUIDS_PRIMITIVE)

    def _set_onlyconsvalid(self):
        fluids_setcacheinvalid(self._c, FLUIDS_FLAGSALL)
        fluids_setcachevalid(self._c, FLUIDS_CONSERVED)

    property primitive:
        def __set__(self, np.ndarray[np.double_t] P):
            # check P has length 5
            fluids_setattrib(self._c, <double*>P.data, FLUIDS_PRIMITIVE)
        def __get__(self):
            # get length of primitive here
            cdef np.ndarray[np.double_t] P = np.zeros(5)
            self._getattrib(<double*>P.data, FLUIDS_PRIMITIVE)
            return P

    property conserved:
        def __set__(self, np.ndarray[np.double_t] U):
            # check U has length 5
            fluids_setattrib(self._c, <double*>U.data, FLUIDS_CONSERVED)
        def __get__(self):
            # get length of primitive here
            cdef np.ndarray[np.double_t] U = np.zeros(5)
            self._getattrib(<double*>U.data, FLUIDS_CONSERVED)
            return U

    property gammalawindex:
        def __set__(self, double gam):
            fluids_setattrib(self._c, &gam, FLUIDS_GAMMALAWINDEX)
        def __get__(self):
            cdef double gam
            self._getattrib(&gam, FLUIDS_GAMMALAWINDEX)
            return gam

    def eigenvalues(self, dim=0):
        cdef int flag = [FLUIDS_EVAL0, FLUIDS_EVAL1, FLUIDS_EVAL2][dim]
        cdef np.ndarray[np.double_t] L = np.zeros(5)
        self._getattrib(<double*>L.data, flag)
        return L

    def eigenvectors(self, dim=0):
        cdef int flagL = [FLUIDS_LEVECS0, FLUIDS_LEVECS1, FLUIDS_LEVECS2][dim]
        cdef int flagR = [FLUIDS_REVECS0, FLUIDS_REVECS1, FLUIDS_REVECS2][dim]
        cdef np.ndarray[np.double_t,ndim=2] L = np.zeros((5,5))
        cdef np.ndarray[np.double_t,ndim=2] R = np.zeros((5,5))
        self._getattrib(<double*>L.data, flagL)
        self._getattrib(<double*>R.data, flagR)
        return L, R


class state_buffer(np.ndarray):
    """
    A subclass of np.ndarray which invokes a hook on setting elements of the
    array. The hook is sent the list of absolute indices which were modified in
    the set operation, but with considering the whole of the last dimension as a
    single index.
    """
    def __init__(self, *args, **kwargs):
        super(state_buffer, self).__init__(*args, **kwargs)
        self._mask = np.array(range(self.size)) / self.shape[-1]
        self._mask.resize(self.shape)
        self._hook = None

    def __setitem__(self, *args, **kwargs):
        self._hook(np.unique(self._mask[args[0]]))
        super(state_buffer, self).__setitem__(*args, **kwargs)


class FluidStateVector(object):
    """
    Class representing an array of FluidState's. The conserved and primitive
    data of each FluidState are mapped over their own contiguous ndarray's. Set
    operations on those ndarray's trigger the proper data invalidation on the
    corresponding FluidState's.

    WARNING: Care must be taken when the data buffers owned by this instances of
    class are modified directly in C code. In that case, the set hooks will not
    be invoked, and the FluidState will be invalid without knowing it.
    """
    def __init__(self, shape):

        def priminvalid(ind):
            for s in states.flat[ind]:
                s._set_onlyprimvalid()
        def consinvalid(ind):
            for s in states.flat[ind]:
                s._set_onlyconsvalid()

        states = np.ndarray(shape=shape, dtype=FluidState)
        primbuf = state_buffer(tuple(shape) + (5,))
        consbuf = state_buffer(tuple(shape) + (5,))
        primbuf._hook = priminvalid
        consbuf._hook = consinvalid

        for i in range(states.size):
            states.flat[i] = FluidState()
            states.flat[i]._map_bufferp(primbuf, i)
            states.flat[i]._map_bufferc(consbuf, i)

        self._states = states
        self._primbuf = primbuf
        self._consbuf = consbuf


cdef class RiemannSolver(object):
    def __cinit__(self):
        self._c = fluids_riemann_new()
        self.SL = None
        self.SR = None

    def __dealloc__(self):
        fluids_riemann_del(self._c)

    def __init__(self):
        fluids_riemann_setsolver(self._c, FLUIDS_RIEMANN_EXACT)

    property solver:
        def __set__(self, val):
            solvers = {'hll': FLUIDS_RIEMANN_HLL,
                       'hllc': FLUIDS_RIEMANN_HLLC,
                       'exact': FLUIDS_RIEMANN_EXACT,}
            fluids_riemann_setsolver(self._c, solvers[val])

    def set_states(self, FluidState SL, FluidState SR):
        assert SL.gammalawindex == SR.gammalawindex
        self.SL = SL # hold onto these so they're not deleted
        self.SR = SR
        fluids_riemann_setdim(self._c, 0)
        fluids_riemann_setstateL(self._c, SL._c)
        fluids_riemann_setstateR(self._c, SR._c)
        fluids_riemann_execute(self._c)

    def sample(self, double s):
        if self.SL is None or self.SR is None:
            raise ValueError("solver needs a need a left and right state")
        cdef FluidState S = FluidState()
        S.gammalawindex = self.SL.gammalawindex
        fluids_riemann_sample(self._c, S._c, s) # sets S.primitive
        return S
