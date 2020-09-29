#######
backend
#######

****
dark
****

.. automodule:: beagles.backend.dark
	:members:
	:undoc-members:

**
io
**

.. automodule:: beagles.backend.io
	:members:
	:undoc-members:

***
net
***

.. automodule:: beagles.backend.net
	:members:
	:undoc-members:


hyperparameters
---------------

.. automodule:: beagles.backend.net.hyperparameters
	:members:


framework
---------

.. automodule:: beagles.backend.net.framework
	:members:

tfnet
-----
.. automodule:: beagles.backend.net.tfnet
	:members:

####
base
####

.. automodule:: beagles.base
	:members:
	:undoc-members:


#######
scripts
#######

*******
RAMDisk
*******

.. code-block:: bash

	RAMDisk_mount() {
	...
	}

	RAMDisk_unmount() {
	...
	}

	if [[ "$1" = "mount" ]]; then
		RAMDisk_mount
	elif [[ "$1" = "unmount" ]]; then
		RAMDisk_unmount
	else
		>&2 echo "line $LINENO: $NAME_$1: command not found"
	fi

##
ui
##

.. automodule:: beagles.ui
	:members:
	:undoc-members:

