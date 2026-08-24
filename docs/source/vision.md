# tensorplay.vision

`tensorplay.vision` is the computer-vision toolkit of TensorPlay, mirroring
the `torchvision` package layout. The listing below is a static overview; the
package requires optional dependencies (Pillow) for some entry points, so this
page intentionally does not use autodoc.

## Top-level functions

- `to_tensor`, `from_image`, `from_file` — tensor/image conversion
- `make_grid`, `save_image` — visualization helpers
- `set_backend`, `get_backend` — image backend selection (PIL)

## Submodules

- {mod}`tensorplay.vision.datasets` — dataset classes and utilities
  (`MNIST`, `CIFAR10`/`CIFAR100`, `ImageFolder`, `Folder`,
  `DatasetFolder`, `UCF101`, ...)
- {mod}`tensorplay.vision.transforms` — composable transforms
  (`Compose`, `ToTensor`, `Normalize`, `Resize`, `CenterCrop`,
  `RandomCrop`, `RandomHorizontalFlip`, ...)
- {mod}`tensorplay.vision.models` — reference model architectures
  (`AlexNet`, `VGG`, `ResNet` variants, ...)
- {mod}`tensorplay.vision.io` — image reading/writing backends
- {mod}`tensorplay.vision.ops` — vision-specific operators
- {mod}`tensorplay.vision.utils` — helpers shared across the package
