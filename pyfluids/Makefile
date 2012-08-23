

.PHONY: PYFLUIDS

default : PYFLUIDS

PYFLUIDS : 
	python setup.py build_ext --inplace

clean :
	python setup.py clean --all
	rm -f *.so *.c *.pyc
