# ex: set ts=8 noet:

all: qt5

test: testpy3

testpy3:
	python3 -m unittest discover tests

qt5: qt5py3

qt5py3:
	pyrcc5 -o libs/resources.py resources.qrc
	python3 setup.py build_ext --inplace

clean:
	rm -f ~/.SLGR-SuiteSettings.pkl ./resources/resources.py
	rm -f ./libs/cython_utils/*.c
	rm -f ./libs/cython_utils/*.so
	rm -rf ./build

.PHONY: test
