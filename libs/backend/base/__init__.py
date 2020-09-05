from __future__ import annotations
from typing import Union, AnyStr, List, Type, Any
from abc import abstractmethod


class Subsystem(object):
    token = dict()

    @abstractmethod
    def constructor(self, *args):
        raise NotImplementedError

    class __metaclass__(type):
        msg = 'Subsystem objects should have a constructor() not an __init__().'

        def __init__(cls, *args, **kw):
            super(type, cls).__init__(*args, **kw)
            if cls.__init__ is not Subsystem.__init__:
                raise NotImplementedError(cls.msg)


class SubsystemFactory(object):
    msg = "SubsystemFactory subclasses must define a create classmethod"
    create_key = object()
    constructor = Subsystem.constructor
    token = Subsystem.token

    def __call__(self, *args, **kwargs):
        pass

    def __init__(self, create_key, *args):
        if not create_key == self.create_key:
            raise NotImplementedError(self.msg)
        self.constructor(*args)

    @classmethod
    @abstractmethod
    def create(cls, *args) -> SubsystemFactory(Subsystem):
        raise NotImplementedError(cls.msg)


def register_subsystem(token: Union[AnyStr, List], factory: Type[SubsystemFactory]) -> Any:
    """Decorator to register Subsystem metadata tokens to a SubsystemFactory"""

    def deco(cls: Subsystem) -> Subsystem:
        types = token.split(' ')
        multi = dict(zip(types, list([cls]) * len(types)))
        single = {token: cls}
        cls.token = single if len(types) == 1 else multi
        cls.__bases__ = (factory, )
        return cls

    return deco
