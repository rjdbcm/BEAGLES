#######
backend
#######

*******
darknet
*******

darknet
-------

.. automodule:: beagles.backend.darknet.darknet
	:members:
	:inherited-members:
	:undoc-members:
	:exclude-members: darkops

.. autodata:: darkops
	:annotation:

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


augmentation
------------

.. automodule:: beagles.backend.net.augmentation
	:members:


framework
---------

.. automodule:: beagles.backend.net.framework
	:members:


hyperparameters
---------------

.. automodule:: beagles.backend.net.hyperparameters
	:members:


####
base
####

.. automodule:: beagles.base
	:members:
	:undoc-members:

##
io
##

.. automodule:: beagles.io
	:members:
	:undoc-members:

#########
resources
#########

Files to bundle into compiled resources using pyrcc5 from the `Qt resource system <https://doc.qt.io/qt-5/resources.html>`_.

.. _resources.qrc:

*************
resources.qrc
*************

Manifest of all resource files to bundle.
Icons are aliased to action names and strings are aliased as their file basename.

.. literalinclude:: ../beagles/resources/resources.qrc
	:language: xml

*******
actions
*******

Each action will have an entry in :ref:`actions.json`, an entry of the same
name for the main string :ref:`strings.properties`, and an entry in
:ref:`resources.qrc` that points to the icon for the action contained in
:file:`resources/icons`.

.. _actions.json:

actions.json
------------

Serialized partial arguments for the :meth:`beagles.ui.newAction` constructor,
each entry contains a keyboard shortcut, a boolean whether the action is enabled,
and a boolean whether the action is checkable.

.. literalinclude:: ../beagles/resources/actions/actions.json
	:language: json

*******
strings
*******

String resources and localized translations for Qt

.. _strings.properties:

strings.properties
------------------

.. literalinclude:: ../beagles/resources/strings/strings.properties


#######
scripts
#######

.. _ramdisk-ref:

*******
RAMDisk
*******

MacOS-specific tool used by :class:`beagles.io.SharedMemory` to create a shared memory drive.

.. literalinclude:: ../beagles/scripts/RAMDisk

##
ui
##

.. automodule:: beagles.ui
	:members:
	:special-members: __init__, __ior__
	:undoc-members:

*********
callbacks
*********

.. automodule:: beagles.ui.callbacks
	:members:
	:undoc-members:

*********
functions
*********

.. automodule:: beagles.ui.functions
	:members:
	:undoc-members:

*******
widgets
*******

.. automodule:: beagles.ui.widgets
	:members:
	:undoc-members:

