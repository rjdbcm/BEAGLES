
all: virtualenv qt5 cython

dev_package: qt5 local_package

local_package:
	pip3 install -e .

virtualenv:
	virtualenv --python=python3 .
	. bin/activate

qt5:
	pyrcc5 -o beagles/resources.py beagles/resources/resources.qrc

cython:
	python3 setup.py build_ext --inplace

install:
	python3 setup.py install

test:
	python3 -m unittest discover ./tests

distclean: clean clean_site_packages

docs:
	cd ./docs && make html

coverage:
	coverage run -m unittest discover tests

diagrams:
	pyreverse -ASmy -k -o png -p BEAGLES ./beagles

clean:
	find . -name “.DS_Store” -depth -exec rm {} \;
	rm -f ~/.BEAGLESSettings.json ./libs/resources.py
	rm -rf ./.pytest_cache
	rm -f ./libs/backend/net/frameworks/extensions/*.c
	rm -rf *.egg-info
	rm -rf BEAGLES-*
	rm -f ./libs/cythonUtils/*.so
	rm -rf ./build
	rm -rf ./dist
	rm -rf ./bin
	rm -f .Python

clean_site_packages:
	rm -rf ./lib

.PHONY: docs

.PHONY: test

.PHONY: coverage

.PHONY: virtualenv
