"""Base classes and methods for other BEAGLES modules"""
from typing import Union, AnyStr, List, Type, Callable
from abc import abstractmethod
from beagles.base.box import *
from beagles.base.constants import *
from beagles.base.errors import *
from beagles.base.flags import *
from beagles.base.shape import *
from beagles.base.stringBundle import *
import beagles.base.version

PreprocessedBox = PreprocessedBox
"""
:class:`PreprocessedBox`
"""

ProcessedBox = ProcessedBox
"""
:class:`ProcessedBox`
"""

PostprocessedBox = PostprocessedBox
"""
:class:`PostprocessedBox`
"""

Flags = Flags
"""
:class:`Flags`
"""

Shape = Shape
"""
:class:`Shape`
"""

StringBundle = StringBundle
"""
:class:`StringBundle`
"""

getStr = getStr
"""Convenience function to grab strings from :class:`StringBundle`

    Args:
        strId: resource ID for the string to get

    Returns:
        str matching the resource ID
"""


class Subsystem(object):
    token: dict

    class __metaclass__(type):
        msg = 'Subsystem objects should have a constructor() not an __init__().'

        def __init__(cls, *args, **kwargs):
            super(type, cls).__init__(*args, **kwargs)
            if cls.__init__ is not Subsystem.__init__:
                raise NotImplementedError(cls.msg)

    @abstractmethod
    def constructor(self, *args, **kwargs):
        raise NotImplementedError


class SubsystemPrototype(object):
    create_key = object()
    constructor: Subsystem.constructor
    token: dict

    def __init__(self, create_key, *args, **kwargs):
        msg = "SubsystemPrototype subclasses must define a create classmethod"
        if not create_key == self.create_key:
            raise NotImplementedError(msg)
        self.constructor(*args, **kwargs)

    @classmethod
    def get_register(cls) -> dict:
        token_register = dict()
        for subclass in cls.__subclasses__():
            token_register.update(subclass.token)
        return token_register

    @classmethod
    @abstractmethod
    def create(cls, *args):
        raise NotImplementedError(cls.msg)


def register_subsystem(token: Union[AnyStr, List],
                       prototype: Type[SubsystemPrototype]) -> Callable:
    """Decorator to register :class:`Subsystem` metadata tokens to a :class:`SubsystemPrototype`

    Example:
        Can be used with a single text token...

        .. code-block:: python

            @register_subsystem(token='[detection]', prototype=Framework)
            class Yolo(Subsystem):
            ...

        .. code-block:: pycon

            >>> Yolo.token
            {'[detection]': Yolo}

        Or can be be used with multiple token splitting on space...

        .. code-block:: python

            @register_subsystem(token='sse l1 l2 smooth sparse softmax', prototype=Framework)
            class MultiLayerPerceptron(Subsystem):
                ...

        .. code-block:: pycon

            >>> MultiLayerPerceptron.token
            {'sse': MultiLayerPerceptron, l1: MultiLayerPerceptron, l2: MultiLayerPerceptron, ...}

    Returns:
        A registered :class:`Subsystem` with :attr:`Subsystem.token` set to `{cls: token}`
        and it's :attr:`__mro__` overridden with `prototype`.

    Raises:
        TypeError: If :class:`Subsystem` or :class:`SubsystemPrototype` isn't in the registered class MRO

    """

    def raise_error(klass, name, mro):
        msg = f'{klass} not found in {name} MRO: {mro}'
        raise TypeError(msg)

    if SubsystemPrototype not in prototype.__mro__:
        raise_error(SubsystemPrototype, prototype.__name__, prototype.__mro__)

    def decorator(cls: type(Subsystem)) -> Subsystem:
        error = True if Subsystem not in cls.__mro__ else False
        if error and SubsystemPrototype not in cls.__mro__:
            raise_error(Subsystem, cls.__name__, cls.__mro__)
        types = token.split(' ')
        multi = dict(zip(types, list([cls]) * len(types)))
        single = {token: cls}
        cls.token = single if len(types) == 1 else multi
        cls.__bases__ = (prototype,)
        return cls

    return decorator
