
cimport numpy as np

cdef extern from "fluids.h":
    cdef int FLUIDS_LOCATION          = (1<<0)
    cdef int FLUIDS_PASSIVE           = (1<<1)
    cdef int FLUIDS_CONSERVED         = (1<<2)
    cdef int FLUIDS_PRIMITIVE         = (1<<3)
    cdef int FLUIDS_MAGNETIC          = (1<<4)
    cdef int FLUIDS_FOURVELOCITY      = (1<<5)
    cdef int FLUIDS_FLUX0             = (1<<6)
    cdef int FLUIDS_FLUX1             = (1<<7)
    cdef int FLUIDS_FLUX2             = (1<<8)
    cdef int FLUIDS_EVAL0             = (1<<9)
    cdef int FLUIDS_EVAL1             = (1<<10)
    cdef int FLUIDS_EVAL2             = (1<<11)
    cdef int FLUIDS_LEVECS0           = (1<<12)
    cdef int FLUIDS_LEVECS1           = (1<<13)
    cdef int FLUIDS_LEVECS2           = (1<<14)
    cdef int FLUIDS_REVECS0           = (1<<15)
    cdef int FLUIDS_REVECS1           = (1<<16)
    cdef int FLUIDS_REVECS2           = (1<<17)
    cdef int FLUIDS_JACOBIAN0         = (1<<18)
    cdef int FLUIDS_JACOBIAN1         = (1<<19)
    cdef int FLUIDS_JACOBIAN2         = (1<<20)
    cdef int FLUIDS_SOUNDSPEEDSQUARED = (1<<21)
    cdef int FLUIDS_TEMPERATURE       = (1<<22)
    cdef int FLUIDS_SPECIFICENTHALPY  = (1<<23)
    cdef int FLUIDS_SPECIFICINTERNAL  = (1<<24)
    cdef int FLUIDS_GAMMALAWINDEX     = (1<<25)
    cdef int FLUIDS_FLAGSALL          = ((1<<30) - 1)

    cdef int FLUIDS_FLUXALL           = (FLUIDS_FLUX0|FLUIDS_FLUX1|FLUIDS_FLUX2)
    cdef int FLUIDS_EVALSALL          = (FLUIDS_EVAL0|FLUIDS_EVAL1|FLUIDS_EVAL2)
    cdef int FLUIDS_LEVECSALL         = (FLUIDS_LEVECS0|FLUIDS_LEVECS1|FLUIDS_LEVECS2)
    cdef int FLUIDS_REVECSALL         = (FLUIDS_REVECS0|FLUIDS_REVECS1|FLUIDS_REVECS2)
    cdef int FLUIDS_JACOBIANALL       = (FLUIDS_JACOBIAN0|FLUIDS_JACOBIAN1|FLUIDS_JACOBIAN2)

    cdef int FLUIDS_SCALAR_ADVECTION  = -41
    cdef int FLUIDS_SCALAR_BURGERS    = -42
    cdef int FLUIDS_SHALLOW_WATER     = -43
    cdef int FLUIDS_NRHYD             = -44
    cdef int FLUIDS_SRHYD             = -45
    cdef int FLUIDS_URHYD             = -46
    cdef int FLUIDS_GRHYD             = -47
    cdef int FLUIDS_NRMHD             = -48
    cdef int FLUIDS_SRMHD             = -49
    cdef int FLUIDS_GRMHD             = -50

    cdef int FLUIDS_EOS_GAMMALAW      = -51
    cdef int FLUIDS_EOS_TABULATED     = -52

    cdef int FLUIDS_COORD_CARTESIAN   = -53
    cdef int FLUIDS_COORD_SPHERICAL   = -54
    cdef int FLUIDS_COORD_CYLINDRICAL = -55

    cdef int FLUIDS_ERROR_BADARG      = -66
    cdef int FLUIDS_ERROR_BADREQUEST  = -67
    cdef int FLUIDS_ERROR_RIEMANN     = -68
    cdef int FLUIDS_ERROR_INCOMPLETE  = -69

    cdef int FLUIDS_RIEMANN_HLL       = -70
    cdef int FLUIDS_RIEMANN_HLLC      = -71
    cdef int FLUIDS_RIEMANN_EXACT     = -72

    struct fluid_state
    struct fluid_riemann

    fluid_state *fluids_new()
    int fluids_del(fluid_state *S)
    int fluids_update(fluid_state *S, long flags)
    int fluids_setcachevalid(fluid_state *S, long flags)
    int fluids_setcacheinvalid(fluid_state *S, long flags)
    int fluids_getlastupdate(fluid_state *S, long *flags)
    int fluids_alloc(fluid_state *S, long flags)
    int fluids_dealloc(fluid_state *S, long flags)
    int fluids_mapbuffer(fluid_state *S, long flag, void *buffer)
    int fluids_setfluid(fluid_state *S, int fluid)
    int fluids_seteos(fluid_state *S, int eos)
    int fluids_setcoordsystem(fluid_state *S, int coordsystem)
    int fluids_setnpassive(fluid_state *S, int n)
    int fluids_getattrib(fluid_state *S, double *x, long flag)
    int fluids_setattrib(fluid_state *S, double *x, long flag)
    int fluids_getnwaves(int fluid)

    fluid_riemann *fluids_riemann_new()
    int fluids_riemann_del(fluid_riemann *R)
    int fluids_riemann_setstateL(fluid_riemann *R, fluid_state *S)
    int fluids_riemann_setstateR(fluid_riemann *R, fluid_state *S)
    int fluids_riemann_setdim(fluid_riemann *R, int dim)
    int fluids_riemann_execute(fluid_riemann *R)
    int fluids_riemann_sample(fluid_riemann *R, fluid_state *S, double s)
    int fluids_riemann_setsolver(fluid_riemann *R, int solver)


cdef class FluidState(object):
    cdef fluid_state *_c
    cdef _getattrib(self, double *val, int flag)
    cpdef _map_bufferp(self, np.ndarray buf, int absindex)
    cpdef _map_bufferc(self, np.ndarray buf, int absindex)
    cdef list _buffers

cdef class RiemannSolver(object):
    cdef fluid_riemann *_c
    cdef FluidState SL
    cdef FluidState SR

