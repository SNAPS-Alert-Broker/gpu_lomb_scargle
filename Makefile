
NGPU?=1
NCPU?=16
export NGPU
export NCPU

all: cuda install

install:
	python setup.py install

cuda:
	make -C gpuls/cuda make_python_shared_libs
