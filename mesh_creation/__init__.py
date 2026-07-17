"""Mesh-creation workflows for the Haberkern Lab CRANTb connectome.

`mesh_creation` turns 3D TIFF segmentation images (e.g. from Thermo Fisher
AMIRA) or STL mesh files into [Neuroglancer](https://github.com/google/neuroglancer)
precomputed neuropil mesh datasets, and merges independently meshed datastacks
into a single publishable dataset. It uses
[Igneous](https://github.com/seung-lab/igneous) mesh generation in the backend.

Every workflow is available three ways that share one implementation: a
command-line tool, a plain function, and an object-oriented configuration class.
This page documents the Python API; the CLI is covered in the
[README](https://github.com/yigityargili991/crantconnectome-haberkernlab-meshcreation#readme).

## Installation

```bash
uv sync            # installs the package plus the tiff-to-mesh / merge-datastacks CLIs
# or: python -m pip install .
```

## The public API at a glance

| Task | Function | Object-oriented |
|------|----------|-----------------|
| TIFF/STL → mesh | `tiff_to_mesh` (alias `create_mesh`) | `MeshConverter.run` (alias `convert`) |
| Merge datastacks | `merge_datastacks` | `DatastackMerger.run` (alias `merge`) |
| Replace labels | `replace_labels` | `DatastackMerger.replace` |

## Convert a TIFF or STL to a mesh

`tiff_to_mesh()` (and its exact alias `create_mesh()`) runs the full conversion
and returns the dataset directory:

```python
from mesh_creation import tiff_to_mesh

dataset = tiff_to_mesh(
    "./my_segmentation.tif",
    output_dir="./meshes",
    resolution=(800, 800, 840),
    voxel_offset=(-54, -54, -3),
    labels={1: "ellipsoid_body", 2: "fan_shaped_body"},
)
# dataset == "./meshes/my_segmentation"
```

The object-oriented form separates configuration from execution; `convert()` is
an alias for `run()`:

```python
from mesh_creation import MeshConverter

converter = MeshConverter(
    input_path="./my_segmentation.tif",
    output_dir="./meshes",
    resolution=(800, 800, 840),
    voxel_offset=(-54, -54, -3),
)
dataset = converter.run()
```

Conversion defaults to the sharded Neuroglancer mesh format; pass
`unsharded=True` when unsharded output is wanted. Segment names can come from a
`labels={id: name}` mapping, a `label_file` CSV (`id,name`), or be auto-derived
from TIFF labels / STL filenames.

## Merge datastacks

`merge_datastacks()` combines two or more precomputed datastacks into one
standalone mesh dataset, meshing each source independently so overlapping
structures do not interfere. It safely defaults to `unsharded=True` and returns
(and writes to `label_map.json`) a map from each source to its
`old_id -> new_id` remapping:

```python
from mesh_creation import merge_datastacks

label_map = merge_datastacks(
    ["./stack_A", "./stack_B"],
    "./merged",
    labels={"stack_A": {1: "PB"}},
    exclude={"stack_A": [2], "stack_B": ["obsolete"]},
)
```

Group keys in `labels`, `exclude`, and `include` are absolute source paths or
basenames unique among the inputs; label selectors are numeric IDs or names from
the source's segment properties. The object-oriented equivalent is
`DatastackMerger.run()` (alias `merge()`):

```python
from mesh_creation import DatastackMerger

merger = DatastackMerger(
    datastacks=["./stack_A", "./stack_B"],
    output_dir="./merged",
    label_names={"stack_A": {1: "PB"}},
    exclude={"stack_A": [2]},
)
label_map = merger.run()
```

## Replace labels between two stacks

`replace_labels()` excludes selected labels from a base stack and substitutes
geometry from a replacement stack. By default every non-zero replacement label
is included; `replacement_labels` restricts that:

```python
from mesh_creation import replace_labels

label_map = replace_labels(
    base="./stack_A",
    replacement="./stack_B_new_pb",
    labels=["PB"],
    output_dir="./merged",
    replacement_labels=["PB_left", "PB_right"],  # omit to include all of B
)
```

The object-oriented equivalent takes the base first and replacement second:

```python
from mesh_creation import DatastackMerger

merger = DatastackMerger(datastacks=["./stack_A", "./stack_B_new_pb"], output_dir="./merged")
label_map = merger.replace(labels=["PB"], replacement_labels=["PB_left", "PB_right"])
```

Replacement is a group-level merge, not a voxelwise overwrite: sources are
meshed independently, IDs are not preserved, and all kept and replacement labels
receive the deterministic contiguous remap used by `merge_datastacks()`.

## Adding a mesh to Neuroglancer

After running with `setgit=True`/`--setgit`, push the output to GitHub and add
its raw content URL as a `neuroglancer-precomputed:` source. Passing
`push="repo_name"`/`--push` automates the repo creation, push, and link printing.
The baked-in voxel offset means no manual source transform is required.

---

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
