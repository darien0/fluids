
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
        def __get__(self):
            # get length of primitive here
            cdef np.ndarray[np.double_t] P = np.zeros(5)
            fluids_getattrib(self._c, <double*>P.data, FLUIDS_PRIMITIVE)
            return P

