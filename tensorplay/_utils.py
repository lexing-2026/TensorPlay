
class _ClassPropertyDescriptor:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, instance, owner=None):
        if owner is None:
            owner = type(instance)
        return self.fget.__get__(instance, owner)()

def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return _ClassPropertyDescriptor(func)