import os
from typing import List, Tuple, Dict, Callable, Optional, Iterable
from ...utils.data import Dataset
from ..backend import get_backend, _OPENCV_AVAILABLE

__all__ = ["VisionDataset"]

class VisionDataset(Dataset):
    _repr_indent = 4

    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        head += self.extra_repr()
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        if hasattr(self, "target_transform") and self.target_transform is not None:
            body += [repr(self.target_transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def extra_repr(self) -> str:
        return ""
