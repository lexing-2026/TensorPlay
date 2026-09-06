# tensorplay.audio

`tensorplay.audio` is the audio I/O and processing toolkit of TensorPlay.
The listing below is a static
overview; backend availability depends on installed optional dependencies, so
this page intentionally does not use autodoc.

## Top-level functions

- `load`, `save`, `info` — audio file I/O and metadata
  (`AudioMetaData`)
- `check_available`, `list_audio_backends`, `get_audio_backend`,
  `set_audio_backend` — backend discovery and selection

## Submodules

- {mod}`tensorplay.audio.functional` — functional audio operations
- {mod}`tensorplay.audio.transforms` — composable audio transforms
- {mod}`tensorplay.audio.datasets` — audio datasets (`CMUDict`, ...)
- {mod}`tensorplay.audio.models` — reference audio models
- {mod}`tensorplay.audio.compliance` — standards-compliance transforms
- {mod}`tensorplay.audio.utils` — shared helpers
