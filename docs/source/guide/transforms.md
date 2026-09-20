# Transforms

Data rarely arrives ready for a model. Images are different sizes, pixel values are in a
different range than a network expects, and you want to augment training data with random
flips and crops. Transforms are small callables that turn one piece of data into another, and
`tensorplay.vision.transforms` bundles a standard set.

```python
import tensorplay as tp
import tensorplay.vision.transforms as T
```

## A transform is just a callable

Pass a tensor in, get a tensor out:

```python
img = tp.randn(3, 32, 32)          # a channel-first image (C, H, W)
print(T.Resize((16, 16))(img).shape)   # (3, 16, 16)
```

The channel-first layout `(C, H, W)` is TensorPlay's image convention. Transforms operate on it
directly.

## Compose chains transforms

`Compose` runs a list of transforms in order. This is the standard building block for a
preprocessing or augmentation pipeline:

```python
pipeline = T.Compose([
    T.Resize((28, 28)),
    T.RandomHorizontalFlip(),      # augment: flip with 50% probability
    T.Normalize((0.5,), (0.25,)),  # center and scale each channel
])

img = tp.randn(3, 48, 48)
out = pipeline(img)
print(out.shape)   # (3, 28, 28)
```

## Transforms you will use

- **Shape and size** — `Resize`, `RandomCrop`, `CenterCrop`, `Pad`. These change `(H, W)`.
- **Geometry for augmentation** — `RandomHorizontalFlip`, `RandomVerticalFlip`,
  `RandomRotation`, `RandomAffine`, `RandomPerspective`. These randomly perturb the image on
  each call, which is how you get more varied training data from a fixed dataset.
- **Photometric** — `ColorJitter` (brightness, contrast, saturation, hue together),
  `Grayscale`, `GaussianBlur`, `RandomSolarize`, `RandomPosterize`, `RandomEqualize`,
  `RandomInvert`.
- **Type and range** — `Normalize` (center and scale each channel), `ConvertImageDtype`,
  `ToTensor` (convert a PIL or NumPy image into an int tensor in the `[0, 1]` range).

## Using a transform in a data pipeline

Transforms are usually applied inside a dataset's `__getitem__` so every sample is preprocessed
as the loader yields it:

```python
from tensorplay.utils.data import Dataset, DataLoader
import tensorplay.vision.transforms as T

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = self.images[i]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[i]

images = tp.randn(20, 3, 40, 40)          # N, C, H, W
labels = tp.randint(0, 10, (20,))
dataset = ImageDataset(images, labels, transform=T.Compose([
    T.RandomHorizontalFlip(),
    T.Normalize((0.5,), (0.25,)),
]))
loader = DataLoader(dataset, batch_size=4)
for batch_x, batch_y in loader:
    print(batch_x.shape, batch_y.shape)   # (4, 3, 40, 40) (4,)
    break
```

Augmentation inside `__getitem__` also means each epoch sees a slightly different version of
the data, which generally helps the model generalize.

Note: `ToTensor` converts a `PIL` image or NumPy array (height, width, channels) into a
channel-first tensor. If you are building a pipeline around a dataset that yields tensors, you
can apply `Resize`, `RandomHorizontalFlip`, and `Normalize` directly to them and skip `ToTensor`.

## Where to go next

- The [vision reference](../vision.md) lists every transform and dataset available.
- [Models](models.md) for building the network that consumes these tensors.
