#!/bin/sh

echo Installing platform specific requirements...

if [[ $OSTYPE == "linux-gnu" ]]; then
    sudo apt-get install python3-pip pyqt5-dev-tools libomp-dev build-essential

elif [[ $OSTYPE == "darwin"* ]]; then
    which -s brew
    if [[ $? != 0 ]] ; then
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

    which -s python3
    if [[ $? != 0 ]] ; then
        # Install Python3
        brew install python@3
    fi

else
    echo Error: unknown platform detected!
    exit 1
fi

cd ../
make
