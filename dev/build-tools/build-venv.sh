#!/bin/sh

echo Installing platform specific requirements...
platform=$(uname)


if [ "$platform" = "Linux" ]; then
    sudo apt-get install python3-pip pyqt5-dev-tools libomp-dev build-essential

elif [ "$platform" = "Darwin" ]; then

    if ! command -v brew ; then
        # Install Homebrew
        /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    fi

    if brew ls --versions libxml2 > /dev/null; then
        # Install libxml
        brew install libxml2
    fi

    if brew ls --versions qt > /dev/null; then
        # Install qt
        brew install qt
    fi

    if ! command -v python3 ; then
        # Install Python3
        brew install python@3
    fi

else
    echo Error: unknown platform detected!
    exit 1
fi

echo Compiling and installing BEAGLES in place using virtualenv!

cd ../../
virtualenv --python=python3 .
. bin/activate
pip3 install -r requirements.txt
pyrcc5 -o libs/resources.py resources.qrc
python3 setup.py build_ext --inplace
