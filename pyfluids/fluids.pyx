
cimport fluids
cimport numpy as np
import numpy as np

cdef class FluidState(object):
    def __cinit__(self):
        self._c = fluids_new()

    def __dealloc__(self):
        fluids_del(self._c)

    def __init__(self):
        fluids_setfluid(self._c, FLUIDS_NRHYD)

    property primitive:
        def __set__(self, np.ndarray[np.double_t] P):
            # check P has length 5
            fluids_setattrib(self._c, <double*>P.data, FLUIDS_PRIMITIVE)
            fluids_p2c(self._c)
        def __get__(self):
            # get length of primitive here
            cdef np.ndarray[np.double_t] P = np.zeros(5)
            fluids_getattrib(self._c, <double*>P.data, FLUIDS_PRIMITIVE)
            return P

    property conserved:
        def __set__(self, np.ndarray[np.double_t] U):
            # check U has length 5
            fluids_setattrib(self._c, <double*>U.data, FLUIDS_CONSERVED)
            fluids_c2p(self._c)
        def __get__(self):
            # get length of primitive here
            cdef np.ndarray[np.double_t] U = np.zeros(5)
            fluids_getattrib(self._c, <double*>U.data, FLUIDS_CONSERVED)
            return U

    property gammalawindex:
        def __set__(self, double gam):
            fluids_setattrib(self._c, &gam, FLUIDS_GAMMALAWINDEX)
        def __get__(self):
            cdef double gam
            fluids_getattrib(self._c, &gam, FLUIDS_GAMMALAWINDEX)
            return gam

    def eigenvalues(self, dim=0):
        cdef int flag = [FLUIDS_EVALS0, FLUIDS_EVALS1, FLUIDS_EVALS2][dim]
        cdef np.ndarray[np.double_t] L = np.zeros(5)
        fluids_update(self._c, flag)
        fluids_getattrib(self._c, <double*>L.data, flag)
        return L

    def eigenvectors(self, dim=0):
        cdef int flagL = [FLUIDS_LEVECS0, FLUIDS_LEVECS1, FLUIDS_LEVECS2][dim]
        cdef int flagR = [FLUIDS_REVECS0, FLUIDS_REVECS1, FLUIDS_REVECS2][dim]
        cdef np.ndarray[np.double_t,ndim=2] L = np.zeros((5,5))
        cdef np.ndarray[np.double_t,ndim=2] R = np.zeros((5,5))
        fluids_update(self._c, flagL | flagR)
        fluids_getattrib(self._c, <double*>L.data, flagL)
        fluids_getattrib(self._c, <double*>R.data, flagR)
        return L, R


cdef class RiemannSolver(object):
    def __cinit__(self):
        self._c = fluids_riemann_new()
        self.SL = None
        self.SR = None

    def __dealloc__(self):
        fluids_riemann_del(self._c)

    def __init__(self):
        pass

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

