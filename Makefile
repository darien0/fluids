

# install prefix (use as command line variable)
PREFIX ?= $(pwd)

# install prefix (use as environment variable)
FLUIDSLIB_INSTALL ?= $(PREFIX)

.PHONY : clib install

default : clib

all : clib

clib : 
	@make -C src

#python :
#	@make -C python

#cython : clib
#	@make -C cython

install : clib
	mkdir -p $(FLUIDSLIB_INSTALL)/include; cp include/* $(FLUIDSLIB_INSTALL)/include
	mkdir -p $(FLUIDSLIB_INSTALL)/bin; cp bin/* $(FLUIDSLIB_INSTALL)/bin
	mkdir -p $(FLUIDSLIB_INSTALL)/lib; cp lib/* $(FLUIDSLIB_INSTALL)/lib

clean :
	@make -C src clean
#	@make -C python clean
#	@make -C cython clean
	@rm -rf lib bin include
