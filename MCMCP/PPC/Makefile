ifdef opcl
opts	=	-I$(opcl)/include -L$(opcl)/lib/x86_64 -lOpenCL
else
opts	=	-framework OpenCL
endif

gcpp	=	$(CXX) ppc.cxx -O2 -ffast-math $(opts)

mlib	=	-fPIC -DXLIB -c -o ppc.o && $(CC) -shared \
		-fPIC -Wl,-soname,xppc ppc.o -o libxppc.so

warn	=	2>&1 | grep -v assuming

all:
	@echo "	make ocl:  compile the ppc  executable for OCL"
	@echo "	make olib: compile the libxppc library for OCL"

oo:
	$(gcpp) -o ppc.o -DXLIB -c

ocl:
	$(gcpp) -o ppc

olib:
	$(gcpp) $(mlib)

clean:
	rm ppc.o ppc libxppc.so || true
