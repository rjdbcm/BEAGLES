# ex: set ts=8 noet:

all: qt5

test: testpy3

qt5:
	pyrcc5 -o libs/resources.py resources.qrc
	python3 setup.py build_ext --inplace

testpy3:
	python3 -m unittest discover tests

clean:
	rm -f ~/.SLGR-SuiteSettings.pkl ./libs/resources.py
	rm -f ./libs/cython_utils/*.c
	rm -rf *.egg-info
	rm -rf SLGR-Suite-*
	rm -f ./libs/cython_utils/*.so
	rm -rf ./build
	rm -rf ./dist

.PHONY: test
