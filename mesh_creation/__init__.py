"""Programmatic API for the CRANT mesh-creation workflows.

The functional and object-oriented entry points share the same implementations:

``tiff_to_mesh(...)`` / ``MeshConverter(...).run()``
    Convert TIFF or STL inputs to a Neuroglancer precomputed mesh dataset.

``merge_datastacks(...)`` / ``DatastackMerger(...).run()``
    Merge independently meshed datastacks.

``replace_labels(...)`` / ``DatastackMerger(...).replace(...)``
    Exclude selected labels from a base stack and substitute geometry from a
    replacement stack.
"""

from .conversion import (
    DEFAULT_RESOLUTION,
    DEFAULT_VOXEL_OFFSET,
    MeshConverter,
    MeshEntryLabels,
    build_parser as build_tiff_parser,
    create_mesh,
    ensure_uint32_labels,
    tiff_to_mesh,
)
from .merging import (
    DatastackMerger,
    build_parser as build_merge_parser,
    merge_datastacks,
    replace_datastack_labels,
    replace_labels,
)

__version__ = "0.6.0"

__all__ = [
    "DEFAULT_RESOLUTION",
    "DEFAULT_VOXEL_OFFSET",
    "DatastackMerger",
    "MeshConverter",
    "MeshEntryLabels",
    "build_merge_parser",
    "build_tiff_parser",
    "create_mesh",
    "ensure_uint32_labels",
    "merge_datastacks",
    "replace_datastack_labels",
    "replace_labels",
    "tiff_to_mesh",
]
