# ex: set ts=8 noet:

all: qt4

test: testpy3

testpy3:
	python3 -m unittest discover tests

qt4: qt4py3

qt5: qt5py3

qt4py3:
	pyrcc4 -py3 -o resources.py resources.qrc
	cd ./backend;   python3 setup.py build_ext --inplace

qt5py3:
	pyrcc5 -o resources.py resources.qrc
	cd ./backend;   python3 setup.py build_ext --inplace

clean:
	rm -f ~/.labelImgSettings.pkl resources.pyc
	rm -f ./backend/darkflow/cython_utils/*.c
	rm -f ./backend/darkflow/cython_utils/*.so
	rm -rf ./backend/build
	rm -rf ./build

.PHONY: test
