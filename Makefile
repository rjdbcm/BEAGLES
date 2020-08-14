# ex: set ts=8 noet:

all: virtualenv qt5 install

dev_package: qt5 local_package

local_package:
	pip3 install -e .

virtualenv:
	virtualenv --python=python3 .
	. bin/activate

qt5:
	pyrcc5 -o libs/resources.py resources.qrc

cython:
	python3 setup.py build_ext --inplace

install:
	python3 setup.py install

test:
	python3 -m unittest discover ./tests

distclean: clean clean_site_packages

coverage:
	coverage run -m unittest discover tests

clean:
	rm -f ~/.BEAGLESSettings.json ./libs/resources.py
	rm -rf ./.pytest_cache
	rm -f ./libs/cythonUtils/*.c
	rm -rf *.egg-info
	rm -rf BEAGLES-*
	rm -f ./libs/cythonUtils/*.so
	rm -rf ./build
	rm -rf ./dist
	rm -rf ./bin
	rm -f .Python

clean_site_packages:
	rm -rf ./lib

.PHONY: test

.PHONY: coverage

.PHONY: virtualenv
