import os
import numpy as np
from PIL import Image
from .vision import VisionDataset
from tensorplay.hub import download_url_to_file, get_dir

__all__ = ["MNIST"]

def get_int(b):
    return int.from_bytes(b, 'big')

def read_label_file(path):
    import gzip
    import tensorplay as tp
    with gzip.open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8).copy()
        return tp.tensor(parsed).long()

def read_image_file(path):
    import gzip
    import tensorplay as tp
    with gzip.open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16).copy()
        return tp.tensor(parsed).view(length, num_rows, num_cols)

class MNIST(VisionDataset):
    """
    `MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string, optional): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist. If None, uses default cache directory.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    mirrors = [
        'https://ossci-datasets.s3.amazonaws.com/mnist/',
        'http://yann.lecun.com/exdb/mnist/',
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.tpm'
    test_file = 'test.tpm'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root=None, train=True, transform=None, target_transform=None, download=False):
        if root is None:
            root = get_dir()
        super(MNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        
        # Load data
        import tensorplay as tp
        loaded = tp.load(os.path.join(self.processed_folder, data_file))
        self.data, self.targets = loaded

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # target is a 0-d tensor, we need int for some transforms or just keep it as tensor?
        # Standard MNIST returns int for target usually.
        # But here self.targets[index] returns a 0-d tensor.
        # We need .item() to get python scalar from 0-d tensor.
        if hasattr(target, 'item'):
            target = target.item()
        else:
            target = int(target)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.data.size(0)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'MNIST', 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            fpath = os.path.join(self.raw_folder, filename)
            # Try mirrors
            downloaded = False
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    print(f"Downloading {url}")
                    # download_url_to_file handles hash check and skipping if exists
                    download_url_to_file(url, fpath, hash_prefix=md5)
                    downloaded = True
                    break
                except Exception as e:
                    print(f"Error downloading {url}: {e}")
                    continue
            
            if not downloaded:
                 raise RuntimeError(f"Failed to download {filename} from any mirror")

        # process and save as torch files
        print('Processing...')
        
        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte.gz')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte.gz'))
        )
        
        import tensorplay as tp
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            tp.save(training_set, f)
        
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte.gz')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte.gz'))
        )
        
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            tp.save(test_set, f)

        print('Done!')
