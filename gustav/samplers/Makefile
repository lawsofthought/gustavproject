modules = random.mod
objects = random.o utils.o

random.o random.mod:
	gfortran -fPIC -c random.f95

utils.o: $(modules)
	gfortran -g -fbounds-check -fbacktrace -finit-real=nan -fPIC -c utils.f95

samplers: $(modules) $(objects)
	f2py2 --f90flags="-fbacktrace -ffree-form" --fcompiler=gnu95 -c -m fortransamplers $(objects) samplers.f95

# Make a Python external module of utils for testing
utils: $(modules) $(objects)
	f2py2 --f90flags="-ffree-form" --fcompiler=gnu95 -c -m fortranutils random.o utils.f95
	
clean:
	rm -f $(objects) $(modules)

mrproper:
	rm -f  $(objects) $(modules) a.out *.so *.pyc 

all: samplers utils clean
