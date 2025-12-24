import os
from typing import List, Tuple, Dict, Callable, Optional, Iterable
from .vision import VisionDataset
from ..backend import get_backend, _OPENCV_AVAILABLE

__all__ = ["DatasetFolder", "ImageFolder", "make_dataset", "default_loader"]

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def has_file_allowed_extension(filename: str, extensions: Iterable[str]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (str): path to a file
        extensions (Iterable[str]): a list of extensions (case-insensitive)

    Returns:
        bool: True if the filename ends with one of the given extensions.
    """
    return filename.lower().endswith(tuple(ext.lower() for ext in extensions))


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Iterable[str]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None
) -> List[Tuple[str, int]]:
    """
    Generates a list of samples in the form (path_to_sample, class_index).

    Args:
        directory (str): root directory path
        class_to_idx (Dict[str, int]): A mapping from class name to class index.
        extensions (Optional[Iterable[str]]): A list of allowed extensions (case-insensitive).
        is_valid_file (Optional[Callable[[str], bool]]): A function that takes a file path and returns True if the file is valid.

    Returns:
        List[Tuple[str, int]]: includes (path_to_sample, class_index) for valid samples.
    """
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"directory not found: {directory}")
    if is_valid_file is not None:
        validate_file = is_valid_file
    elif extensions is not None:
        validate_file = lambda x: has_file_allowed_extension(x, extensions)
    else:
        raise ValueError("extensions or is_valid_file must be specified to validate files")
    instances: List[Tuple[str, int]] = []
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        try:
            for root, _, names in os.walk(target_dir, followlinks=True):
                for name in sorted(names):
                    path = os.path.join(root, name)
                    if os.path.isdir(path):
                        continue
                    if validate_file(path):
                        instances.append((path, class_index))
        except PermissionError:
            print(f"[warning]: no permission to access directory {target_dir}, skip")
            continue
    return instances


class DatasetFolder(VisionDataset):
    def __init__(self, root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def _find_classes(dir_):
        classes = [d.name for d in os.scandir(dir_) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

def pil_loader(path):
    from PIL import Image
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def opencv_loader(path):
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def default_loader(path):
    backend = get_backend()
    if (backend == "OPENCV" or backend == "ALBUMENTATIONS") and _OPENCV_AVAILABLE:
        return opencv_loader(path)
    return pil_loader(path)


class ImageFolder(DatasetFolder):
    r"""
    A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
    """
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, is_valid_file=None):
        super().__init__(
            root=root,
            loader=loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file
        )
        self.imgs = self.samples
