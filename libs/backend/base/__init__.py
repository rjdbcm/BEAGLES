from __future__ import annotations
from typing import Union, AnyStr, List, Type, Callable
from abc import abstractmethod, ABC


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
    msg = "SubsystemPrototype subclasses must define a create classmethod"
    create_key = object()
    constructor: Subsystem.constructor
    token: dict

    def __init__(self, create_key, *args, **kwargs):
        if not create_key == self.create_key:
            raise NotImplementedError(self.msg)
        self.constructor(*args, **kwargs)

    @classmethod
    def get_register(cls) -> dict:
        token_register = dict()
        for subclass in cls.__subclasses__():
            token_register.update(subclass.token)
        return token_register

    @classmethod
    @abstractmethod
    def create(cls, *args) -> SubsystemPrototype(Subsystem):
        raise NotImplementedError(cls.msg)


def register_subsystem(token: Union[AnyStr, List], prototype: Type[SubsystemPrototype]) -> Callable:
    """Decorator to register Subsystem metadata tokens to a SubsystemPrototype"""

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
