"""Backward-compatible launcher for :mod:`mesh_creation.merging`."""

import os

from mesh_creation.merging import (
    DatastackMerger,
    build_parser,
    main,
    merge_datastacks as _merge_datastacks,
    replace_datastack_labels,
    replace_labels,
)
from shared import MESH_DIR


def _expand_legacy_groups(grouped, source_paths):
    """Re-key source-grouped options by absolute path, accepting basenames too."""
    if grouped is None:
        return None
    expanded = {}
    for source_path in source_paths:
        if source_path in grouped:
            expanded[source_path] = grouped[source_path]
        elif os.path.basename(source_path) in grouped:
            expanded[source_path] = grouped[os.path.basename(source_path)]
    return expanded


def merge_datastacks(
    datastack_dirs,
    output_dir,
    mesh_dir=MESH_DIR,
    unsharded=False,
    manual_labels=None,
    exclusions=None,
    source_properties=None,
):
    """Legacy positional-argument wrapper for :func:`mesh_creation.merge_datastacks`."""
    source_paths = [os.path.abspath(os.fspath(path)) for path in datastack_dirs]
    return _merge_datastacks(
        source_paths,
        output_dir,
        mesh_dir=mesh_dir,
        unsharded=unsharded,
        labels=_expand_legacy_groups(manual_labels, source_paths),
        exclude=_expand_legacy_groups(exclusions, source_paths),
        source_properties=_expand_legacy_groups(source_properties, source_paths),
        validate_labels=False,
    )


__all__ = [
    "DatastackMerger",
    "build_parser",
    "main",
    "merge_datastacks",
    "replace_datastack_labels",
    "replace_labels",
]


if __name__ == "__main__":
    main()
