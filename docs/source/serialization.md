# tensorplay.serialization

## Classes

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.serialization.LoadEndianness
    tensorplay.serialization.StorageType
    tensorplay.serialization.safe_globals
    tensorplay.serialization.set_default_mmap_options
    tensorplay.serialization.skip_data
```

## Functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.serialization.add_safe_globals
    tensorplay.serialization.clear_safe_globals
    tensorplay.serialization.convert_model
    tensorplay.serialization.convert_to_mega
    tensorplay.serialization.default_restore_location
    tensorplay.serialization.get_crc32_options
    tensorplay.serialization.get_default_load_endianness
    tensorplay.serialization.get_default_mmap_options
    tensorplay.serialization.get_safe_globals
    tensorplay.serialization.get_unsafe_globals_in_checkpoint
    tensorplay.serialization.inspect_checkpoint
    tensorplay.serialization.load
    tensorplay.serialization.location_tag
    tensorplay.serialization.normalize_storage_type
    tensorplay.serialization.parse_mega_header
    tensorplay.serialization.register_package
    tensorplay.serialization.resolve_map_location
    tensorplay.serialization.save
    tensorplay.serialization.set_crc32_options
    tensorplay.serialization.set_default_load_endianness
    tensorplay.serialization.storage_to_tensor_type
```

## Format Constants

The serializer identifies its files with a magic number and a protocol
version in the header. Payloads are pickled with
`tensorplay.serialization.DEFAULT_PROTOCOL` unless `save(..., pickle_protocol=...)`
says otherwise, and storages inside mega archives are aligned to
`tensorplay.serialization.DEFAULT_ALIGNMENT` bytes unless `save(..., alignment=...)`
says otherwise. A mega archive is a directory whose name ends with
`tensorplay.serialization.MEGA_EXTENSION`, holding an index file ending with
`tensorplay.serialization.MEGA_INDEX_SUFFIX` next to the serialized data.

- `tensorplay.serialization.MAGIC_NUMBER` — the magic value every archive
  starts with; loaders reject files whose header does not match it.
- `tensorplay.serialization.PROTOCOL_VERSION` — the header protocol version.
- `tensorplay.serialization.DEFAULT_PROTOCOL` — default pickle protocol for
  the payload.
- `tensorplay.serialization.DEFAULT_ALIGNMENT` — default byte alignment for
  storages written into mega archives.
- `tensorplay.serialization.MEGA_EXTENSION` — file suffix of a mega archive.
- `tensorplay.serialization.MEGA_INDEX_SUFFIX` — file suffix of a mega
  archive's index document.
