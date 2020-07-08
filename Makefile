# ex: set ts=8 noet:

all: virtualenv qt5

virtualenv:
	virtualenv --python=python3 .
	. bin/activate
	pip3 install -r requirements/requirements-linux.txt

qt5:
	pyrcc5 -o libs/resources.py resources.qrc
	python3 setup.py build_ext --inplace

test:
	python3 -m unittest discover ./tests

distclean: clean clean_site_packages

clean:
	rm -f ~/.SLGR-SuiteSettings.pkl ./libs/resources.py
	rm -f ./libs/cython_utils/*.c
	rm -rf *.egg-info
	rm -rf SLGR-Suite-*
	rm -f ./libs/cython_utils/*.so
	rm -rf ./build
	rm -rf ./dist
	rm -rf ./bin
	rm -f .Python

clean_site_packages:
	rm -rf ./lib

.PHONY: test

.PHONY: virtualenv
