
cimport numpy as np

cdef extern from "fluids.h":
    int FLUIDS_PRIMITIVE         = (1<<1)
    int FLUIDS_PASSIVE           = (1<<2)
    int FLUIDS_GRAVITY           = (1<<3)
    int FLUIDS_MAGNETIC          = (1<<4)
    int FLUIDS_LOCATION          = (1<<5)
    int FLUIDS_CONSERVED         = (1<<6)
    int FLUIDS_FOURVELOCITY      = (1<<7)
    int FLUIDS_FLUX0             = (1<<8)
    int FLUIDS_FLUX1             = (1<<9)
    int FLUIDS_FLUX2             = (1<<10)
    int FLUIDS_EVAL0             = (1<<11)
    int FLUIDS_EVAL1             = (1<<12)
    int FLUIDS_EVAL2             = (1<<13)
    int FLUIDS_LEVECS0           = (1<<14)
    int FLUIDS_LEVECS1           = (1<<15)
    int FLUIDS_LEVECS2           = (1<<16)
    int FLUIDS_REVECS0           = (1<<17)
    int FLUIDS_REVECS1           = (1<<18)
    int FLUIDS_REVECS2           = (1<<19)
    int FLUIDS_JACOBIAN0         = (1<<20)
    int FLUIDS_JACOBIAN1         = (1<<21)
    int FLUIDS_JACOBIAN2         = (1<<22)
    int FLUIDS_SOUNDSPEEDSQUARED = (1<<23)
    int FLUIDS_TEMPERATURE       = (1<<24)
    int FLUIDS_SPECIFICENTHALPY  = (1<<25)
    int FLUIDS_SPECIFICINTERNAL  = (1<<26)
    int FLUIDS_FLAGSALL          = ((1<<30) - 1)

    int FLUIDS_FLUXALL           = (FLUIDS_FLUX0|FLUIDS_FLUX1|FLUIDS_FLUX2)
    int FLUIDS_EVALSALL          = (FLUIDS_EVAL0|FLUIDS_EVAL1|FLUIDS_EVAL2)
    int FLUIDS_LEVECSALL         = (FLUIDS_LEVECS0|FLUIDS_LEVECS1|FLUIDS_LEVECS2)
    int FLUIDS_REVECSALL         = (FLUIDS_REVECS0|FLUIDS_REVECS1|FLUIDS_REVECS2)
    int FLUIDS_JACOBIANALL       = (FLUIDS_JACOBIAN0|FLUIDS_JACOBIAN1|FLUIDS_JACOBIAN2)

    int FLUIDS_SCADV             = -41 # Scalar advection
    int FLUIDS_SCBRG             = -42 # Burgers equation
    int FLUIDS_SHWAT             = -43 # Shallow water equations
    int FLUIDS_NRHYD             = -44 # Euler equations
    int FLUIDS_SRHYD             = -45 # Special relativistic
    int FLUIDS_URHYD             = -46 # Ultra relativistic
    int FLUIDS_GRHYD             = -47 # General relativistic
    int FLUIDS_NRMHD             = -48 # Magnetohydrodynamic (MHD)
    int FLUIDS_SRMHD             = -49 # Special relativistic MHD
    int FLUIDS_GRMHD             = -50 # General relativistic MHD

    int FLUIDS_EOS_GAMMALAW      = -51
    int FLUIDS_EOS_TABULATED     = -52

    int FLUIDS_COORD_CARTESIAN   = -53
    int FLUIDS_COORD_SPHERICAL   = -54
    int FLUIDS_COORD_CYLINDRICAL = -55

    int FLUIDS_ERROR_BADARG      = -66
    int FLUIDS_ERROR_BADREQUEST  = -67
    int FLUIDS_ERROR_RIEMANN     = -68
    int FLUIDS_ERROR_INCOMPLETE  = -69

    int FLUIDS_RIEMANN_HLL       = -70
    int FLUIDS_RIEMANN_HLLC      = -71
    int FLUIDS_RIEMANN_EXACT     = -72

    int FLUIDS_CACHE_NOTOUCH     = -73
    int FLUIDS_CACHE_RESET       = -74
    int FLUIDS_CACHE_ERASE       = -75

    struct fluids_descr
    struct fluids_cache
    struct fluids_state
    struct fluids_riemn

    # fluids_descr member functions
    fluids_descr *fluids_descr_new()
    int fluids_descr_del(fluids_descr *D)
    int fluids_descr_getfluid(fluids_descr *D, int *fluid)
    int fluids_descr_setfluid(fluids_descr *D, int fluid)
    int fluids_descr_geteos(fluids_descr *D, int *eos)
    int fluids_descr_seteos(fluids_descr *D, int eos)
    int fluids_descr_getcoordsystem(fluids_descr *D, int *coordsystem)
    int fluids_descr_setcoordsystem(fluids_descr *D, int coordsystem)
    int fluids_descr_getgamma(fluids_descr *D, double *gam)
    int fluids_descr_setgamma(fluids_descr *D, double gam)
    int fluids_descr_getncomp(fluids_descr *D, long flag)

    # fluids_state member functions
    fluids_state *fluids_state_new()
    int fluids_state_del(fluids_state *S)
    int fluids_state_setdescr(fluids_state *S, fluids_descr *D)
    int fluids_state_resetcache(fluids_state *S)
    int fluids_state_erasecache(fluids_state *S)
    int fluids_state_getattr(fluids_state *S, double *x, long flag)
    int fluids_state_setattr(fluids_state *S, double *x, long flag)
    int fluids_state_fromcons(fluids_state *S, double *U, int cachebehavior)
    int fluids_state_derive(fluids_state *S, double *x, int flag)

    # fluids_riemn member functions
    fluids_riemn *fluids_riemn_new()
    int fluids_riemn_del(fluids_riemn *R)
    int fluids_riemn_setstateL(fluids_riemn *R, fluids_state *S)
    int fluids_riemn_setstateR(fluids_riemn *R, fluids_state *S)
    int fluids_riemn_setdim(fluids_riemn *R, int dim)
    int fluids_riemn_execute(fluids_riemn *R)
    int fluids_riemn_sample(fluids_riemn *R, fluids_state *S, double s)
    int fluids_riemn_setsolver(fluids_riemn *R, int solver)


cdef class FluidDescriptor(object):
    cdef fluids_descr *_c

cdef class FluidState(object):
    cdef fluids_state *_c
    cdef int _np
    cdef int _ns
    cdef int _ng
    cdef int _nm
    cdef int _nl
    cdef int _disable_cache

"""
cdef class RiemannSolver(object):
    cdef fluids_riemn *_R
    cdef FluidState SL
    cdef FluidState SR
"""
