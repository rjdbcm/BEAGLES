# ex: set ts=8 noet:

all: virtualenv qt5 cython

virtualenv:
	virtualenv --python=python3 .
	. bin/activate
	pip3 install -r requirements/requirements.txt

qt5:
	pyrcc5 -o libs/resources.py resources.qrc

cython:
	python3 setup.py build_ext --inplace

test:
	python3 -m unittest discover ./tests

distclean: clean clean_site_packages

coverage:
	coverage run -m unittest discover tests

clean:
	rm -f ~/.BEAGLESSettings.pkl ./libs/resources.py
	rm -f ./libs/cython_utils/*.c
	rm -rf *.egg-info
	rm -rf BEAGLES-*
	rm -f ./libs/cython_utils/*.so
	rm -rf ./build
	rm -rf ./dist
	rm -rf ./bin
	rm -f .Python

clean_site_packages:
	rm -rf ./lib

.PHONY: test

.PHONY: coverage

.PHONY: virtualenv
