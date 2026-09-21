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

## Random combinators

Three wrappers turn plain transforms into randomized *policies*:

- `RandomApply([t, ...], p)` — run the whole list with probability `p`, pass through
  otherwise.
- `RandomChoice([t1, t2, ...])` — pick one transform at random each call.
- `RandomOrder([t1, t2, ...])` — run all of them, in a fresh random order each call.

```python
augment = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4)], p=0.5)
img = tp.randn(3, 32, 32)
print(augment(img).shape)     # (3, 32, 32) — either jittered or untouched

policy = T.RandomChoice([T.Resize((16, 16)), T.Resize((8, 8))])
print(policy(img).shape)      # (3, 16, 16) or (3, 8, 8), decided per call
```

Combinators nest: `T.RandomApply([T.RandomOrder([...]), ...])` composes the same way
`Compose` does.

## Reproducibility

Random transforms draw from TensorPlay's global RNG, so seeding makes a pipeline
reproducible:

```python
tp.manual_seed(42)
a = T.RandomCrop(4)(tp.randn(3, 8, 8))

tp.manual_seed(42)
b = T.RandomCrop(4)(tp.randn(3, 8, 8))
print(tp.equal(a, b))     # True — same seed, same input, same crop
```

In a multi-worker `DataLoader`, each worker's RNG is seeded deterministically from the
base seed plus the worker id, so augmentations stay random *within* a run but reproducible
*across* runs. See the [randomness note](../notes/randomness.md) for the details.

## Transforms on batches

Geometry transforms such as `Resize` accept a batched `(N, C, H, W)` tensor and map the
same geometry over every image — handy for augmenting a whole batch in one call:

```python
batch = tp.randn(4, 3, 32, 32)
print(T.Resize((16, 16))(batch).shape)    # (4, 3, 16, 16)
```

Random transforms applied to a batch still roll *one* random draw per call, so the whole
batch gets the same flip or crop. If every image should get its own coin flip, apply the
transform per-image inside `__getitem__` — which is the common pattern anyway.

## The functional API

Every transform is a thin object wrapper around a plain function in
`tensorplay.vision.transforms.functional` — `Resize` calls `functional.resize`,
`RandomHorizontalFlip` calls `functional.hflip`, and so on. Use the functions directly when
you want the operation without the randomness or the class machinery:

```python
from tensorplay.vision.transforms import functional as F

img = tp.randn(3, 32, 32)
print(F.resize(img, [16, 16]).shape)     # (3, 16, 16)
```

The functional namespace is also where photometric operations such as `adjust_brightness`,
`adjust_contrast`, `adjust_saturation`, `adjust_hue`, and `autocontrast` live, and where
`InterpolationMode` (the `nearest`/`bilinear`/`bicubic` choice for resizing) is defined.

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
