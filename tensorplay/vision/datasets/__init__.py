"""Datasets, mirroring tensorplay.vision.datasets (video/optical-flow excluded)."""

from .caltech import Caltech101, Caltech256
from .celeba import CelebA
from .cifar import CIFAR10, CIFAR100
from .cityscapes import Cityscapes
from .clevr import CLEVRClassification
from .coco import CocoCaptions, CocoDetection
from .country211 import Country211
from .dtd import DTD
from .eurosat import EuroSAT
from .fakedata import FakeData
from .fer2013 import FER2013
from .fgvc_aircraft import FGVCAircraft
from .flickr import Flickr30k, Flickr8k
from .flowers102 import Flowers102
from .folder import DatasetFolder, ImageFolder
from .food101 import Food101
from .gtsrb import GTSRB
from .imagenet import ImageNet
from .imagenette import Imagenette
from .inaturalist import INaturalist
from .lfw import LFWPairs, LFWPeople
from .lsun import LSUN, LSUNClass
from .mnist import EMNIST, FashionMNIST, KMNIST, MNIST, QMNIST
from .omniglot import Omniglot
from .oxford_iiit_pet import OxfordIIITPet
from .pcam import PCAM
from .phototour import PhotoTour
from .places365 import Places365
from .rendered_sst2 import RenderedSST2
from .sbd import SBDataset
from .sbu import SBU
from .semeion import SEMEION
from .stanford_cars import StanfordCars
from .stl10 import STL10
from .sun397 import SUN397
from .svhn import SVHN
from .usps import USPS
from .vision import VisionDataset
from .voc import VOCDetection, VOCSegmentation
from .widerface import WIDERFace

__all__ = [
    "CIFAR10",
    "CIFAR100",
    "CLEVRClassification",
    "Caltech101",
    "Caltech256",
    "CelebA",
    "Cityscapes",
    "CocoCaptions",
    "CocoDetection",
    "Country211",
    "DTD",
    "DatasetFolder",
    "EMNIST",
    "EuroSAT",
    "FER2013",
    "FGVCAircraft",
    "FakeData",
    "FashionMNIST",
    "Flickr30k",
    "Flickr8k",
    "Flowers102",
    "Food101",
    "GTSRB",
    "INaturalist",
    "ImageFolder",
    "ImageNet",
    "Imagenette",
    "KMNIST",
    "LFWPairs",
    "LFWPeople",
    "LSUN",
    "LSUNClass",
    "MNIST",
    "Omniglot",
    "OxfordIIITPet",
    "PCAM",
    "PhotoTour",
    "Places365",
    "QMNIST",
    "RenderedSST2",
    "SBDataset",
    "SBU",
    "SEMEION",
    "STL10",
    "SUN397",
    "SVHN",
    "StanfordCars",
    "USPS",
    "VOCDetection",
    "VOCSegmentation",
    "VisionDataset",
    "WIDERFace",
]
