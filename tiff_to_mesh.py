"""Backward-compatible launcher for :mod:`mesh_creation.conversion`."""

from mesh_creation.conversion import (
    DEFAULT_RESOLUTION,
    DEFAULT_VOXEL_OFFSET,
    MeshConverter,
    MeshEntryLabels,
    build_parser,
    ensure_uint32_labels,
    main,
    tiff_to_mesh,
)
from shared import get_github_username, push_to_github

__all__ = [
    "DEFAULT_RESOLUTION",
    "DEFAULT_VOXEL_OFFSET",
    "MeshConverter",
    "MeshEntryLabels",
    "build_parser",
    "ensure_uint32_labels",
    "get_github_username",
    "main",
    "push_to_github",
    "tiff_to_mesh",
]


if __name__ == "__main__":
    main()
