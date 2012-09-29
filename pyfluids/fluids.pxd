
cimport numpy as np

cdef extern from "fluids.h":
    enum:
        FLUIDS_PRIMITIVE         =  1<<1,
        FLUIDS_PASSIVE           =  1<<2,
        FLUIDS_GRAVITY           =  1<<3,
        FLUIDS_MAGNETIC          =  1<<4,
        FLUIDS_LOCATION          =  1<<5,
        FLUIDS_CONSERVED         =  1<<6,
        FLUIDS_SOURCETERMS       =  1<<7,
        FLUIDS_FOURVELOCITY      =  1<<8,
        FLUIDS_FLUX0             =  1<<9,
        FLUIDS_FLUX1             =  1<<10,
        FLUIDS_FLUX2             =  1<<11,
        FLUIDS_EVAL0             =  1<<12,
        FLUIDS_EVAL1             =  1<<13,
        FLUIDS_EVAL2             =  1<<14,
        FLUIDS_LEVECS0           =  1<<15,
        FLUIDS_LEVECS1           =  1<<16,
        FLUIDS_LEVECS2           =  1<<17,
        FLUIDS_REVECS0           =  1<<18,
        FLUIDS_REVECS1           =  1<<19,
        FLUIDS_REVECS2           =  1<<20,
        FLUIDS_JACOBIAN0         =  1<<21,
        FLUIDS_JACOBIAN1         =  1<<22,
        FLUIDS_JACOBIAN2         =  1<<23,
        FLUIDS_SOUNDSPEEDSQUARED =  1<<24,
        FLUIDS_TEMPERATURE       =  1<<25,
        FLUIDS_SPECIFICENTHALPY  =  1<<26,
        FLUIDS_SPECIFICINTERNAL  =  1<<27,
        FLUIDS_FLAGSALL          = (1<<30) - 1,
        FLUIDS_FLUXALL     = FLUIDS_FLUX0|FLUIDS_FLUX1|FLUIDS_FLUX2,
        FLUIDS_EVALSALL    = FLUIDS_EVAL0|FLUIDS_EVAL1|FLUIDS_EVAL2,
        FLUIDS_LEVECSALL   = FLUIDS_LEVECS0|FLUIDS_LEVECS1|FLUIDS_LEVECS2,
        FLUIDS_REVECSALL   = FLUIDS_REVECS0|FLUIDS_REVECS1|FLUIDS_REVECS2,
        FLUIDS_JACOBIANALL = FLUIDS_JACOBIAN0|FLUIDS_JACOBIAN1|FLUIDS_JACOBIAN2,

    enum:
        FLUIDS_SUCCESS,
        FLUIDS_ERROR_BADARG,
        FLUIDS_ERROR_BADREQUEST,
        FLUIDS_ERROR_RIEMANN,
        FLUIDS_ERROR_INCOMPLETE,

        FLUIDS_COORD_CARTESIAN,
        FLUIDS_COORD_SPHERICAL,
        FLUIDS_COORD_CYLINDRICAL,

        FLUIDS_SCADV, # Scalar advection
        FLUIDS_SCBRG, # Burgers equation
        FLUIDS_SHWAT, # Shallow water equations
        FLUIDS_NRHYD, # Euler equations
        FLUIDS_GRAVS, # Gravitating Euler equation (with source terms)
        FLUIDS_SRHYD, # Special relativistic
        FLUIDS_URHYD, # Ultra relativistic
        FLUIDS_GRHYD, # General relativistic
        FLUIDS_NRMHD, # Magnetohydrodynamic (MHD)
        FLUIDS_SRMHD, # Special relativistic MHD
        FLUIDS_GRMHD, # General relativistic MHD

        FLUIDS_EOS_GAMMALAW,
        FLUIDS_EOS_TABULATED,

        FLUIDS_RIEMANN_HLL,
        FLUIDS_RIEMANN_HLLC,
        FLUIDS_RIEMANN_EXACT,

        FLUIDS_CACHE_DEFAULT,
        FLUIDS_CACHE_NOTOUCH,
        FLUIDS_CACHE_CREATE,
        FLUIDS_CACHE_STEAL,
        FLUIDS_CACHE_RESET,
        FLUIDS_CACHE_ERASE


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
    int fluids_state_getattr(fluids_state *S, double *x, long flag)
    int fluids_state_setattr(fluids_state *S, double *x, long flag)
    int fluids_state_fromcons(fluids_state *S, double *U, int cache)
    int fluids_state_derive(fluids_state *S, double *x, int flag)
    int fluids_state_cache(fluids_state *S, int operation)


    # fluids_riemn member functions
    fluids_riemn *fluids_riemn_new()
    int fluids_riemn_del(fluids_riemn *R)
    int fluids_riemn_setstateL(fluids_riemn *R, fluids_state *S)
    int fluids_riemn_setstateR(fluids_riemn *R, fluids_state *S)
    int fluids_riemn_setdim(fluids_riemn *R, int dim)
    int fluids_riemn_execute(fluids_riemn *R)
    int fluids_riemn_sample(fluids_riemn *R, fluids_state *S, double s)
    int fluids_riemn_setsolver(fluids_riemn *R, int solver)
    int fluids_riemn_getsolver(fluids_riemn *R, int *solver)


cdef class FluidDescriptor(object):
    cdef fluids_descr *_c


cdef class FluidState(object):
    cdef fluids_state *_c
    cdef FluidDescriptor _descr
    cdef int _np
    cdef int _ns
    cdef int _ng
    cdef int _nm
    cdef int _nl


cdef class FluidStateVector(FluidState):
    cdef np.ndarray _states
    cdef np.ndarray _primitive
    cdef np.ndarray _passive
    cdef np.ndarray _gravity
    cdef np.ndarray _magnetic
    cdef np.ndarray _location


cdef class RiemannSolver(object):
    cdef fluids_riemn *_c
    cdef FluidState SL
    cdef FluidState SR

