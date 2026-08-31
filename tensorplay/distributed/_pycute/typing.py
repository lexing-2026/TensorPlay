from abc import ABC


class Integer(ABC):
    @classmethod
    def __subclasshook__(cls, candidate: type) -> bool:
        if candidate in (bool, float):
            return False
        return issubclass(candidate, int)
