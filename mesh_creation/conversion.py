"""Convert TIFF or STL inputs into a Neuroglancer precomputed mesh dataset.

A TIFF segmentation (one integer label per voxel) or a set of STL surfaces is
loaded into a ``uint32`` label volume, written as a Neuroglancer precomputed
segmentation, and meshed with Igneous into multi-resolution Draco meshes. The
CRANTb-aligned voxel offset is baked into the volume ``info`` so the result
loads at the correct position in Neuroglancer with no manual transform.

Use `tiff_to_mesh` (alias `create_mesh`) for the one-call functional form, or
`MeshConverter` when you want to build the configuration first and run it later.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import tifffile
import trimesh
from cloudvolume import CloudVolume
from taskqueue import LocalTaskQueue

import igneous.task_creation as tc
from shared import (
    MESH_DIR,
    SEGMENT_PROPS_DIR,
    UINT32_MAX,
    AnyPath,
    build_label_names_for_inputs,
    compute_chunk_size,
    parse_label_csv,
    parse_labels,
    push_to_github,
    validate_mesh_dir,
    write_segment_properties,
)

logger = logging.getLogger(__name__)

DEFAULT_RESOLUTION = (800, 800, 840)
"""Default output voxel size in nanometers, as ``(X, Y, Z)``."""

DEFAULT_VOXEL_OFFSET = (-54, -54, -3)
"""Default voxel offset for TIFF inputs, in voxels, for CRANTb atlas alignment.

Baked into the volume ``info`` so the mesh loads in the right place in
Neuroglancer. Pass ``(0, 0, 0)`` for non-CRANTb TIFFs; STL inputs ignore this
and derive their offset from geometry.
"""

SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".stl")
"""Input file extensions accepted by the converter."""

LabelOverrides = Union[Mapping[int, str], Sequence[str]]


def ensure_uint32_labels(array: np.ndarray) -> np.ndarray:
    """Return a ``uint32`` label volume without silently changing label values.

    Segmentation labels must be ``uint32`` for the precomputed format. This
    converts an array to ``uint32`` only when doing so is lossless: booleans and
    in-range integers convert directly, and float arrays convert only if every
    value is finite and integral. Anything that would corrupt label identities
    (a non-integer float, a value outside the ``uint32`` range, an unsupported
    dtype) raises instead of being coerced.

    Args:
        array: Segmentation labels of any integer, boolean, or float dtype.

    Returns:
        The array as ``uint32`` (a view when already ``uint32``, otherwise a
        converted copy).

    Raises:
        ValueError: If a float array holds NaN/inf or non-integer values, or if
            any value falls outside ``[0, UINT32_MAX]``.
        TypeError: If the array dtype is not integer, boolean, or float.
    """
    if array.dtype == np.uint32:
        return array

    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(np.uint32, copy=False)

    if array.size == 0:
        return array.astype(np.uint32, copy=False)

    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError(
                "Input contains NaN or infinite values; cannot convert to uint32 labels safely."
            )
        if not np.equal(array, np.floor(array)).all():
            raise ValueError(
                "Input contains non-integer float values; cannot convert to uint32 labels safely."
            )
        min_val = array.min()
        max_val = array.max()
        if min_val < 0 or max_val > UINT32_MAX:
            raise ValueError(
                f"Input values out of uint32 range [{0}, {UINT32_MAX}]: "
                f"min={min_val}, max={max_val}"
            )
        logger.warning("Converting integer-valued float labels to uint32.")
        return array.astype(np.uint32, copy=False)

    if np.issubdtype(array.dtype, np.integer):
        min_val = array.min()
        max_val = array.max()
        if min_val < 0 or max_val > UINT32_MAX:
            raise ValueError(
                f"Input values out of uint32 range [{0}, {UINT32_MAX}]: "
                f"min={min_val}, max={max_val}"
            )
        return array.astype(np.uint32, copy=False)

    raise TypeError(f"Unsupported dtype for segmentation labels: {array.dtype}")


@dataclass
class MeshEntryLabels:
    """Load TIFF or STL inputs into a uint32 ``(X, Y, Z)`` label volume.

    Construction does the loading: the resulting ``data`` array and
    ``voxel_offset`` are available on the instance once it is created, and
    `build_info` turns them into a Neuroglancer ``info`` dict. A single TIFF
    becomes one labelled volume and uses ``voxel_offset_override`` (default
    ``(0, 0, 0)``); one or more STLs are voxelized into a shared volume with one
    label ID per file, and the offset is derived from their geometry so physical
    position is preserved. TIFF and STL inputs cannot be mixed.

    Attributes:
        file_paths: Input file paths; either exactly one ``.tif``/``.tiff`` or
            one or more ``.stl`` files.
        resolution: Output voxel size in nm, as ``(X, Y, Z)``.
        min_chunks: Lower bound on the number of chunks per axis used to pick
            the precomputed chunk size.
        voxel_offset_override: Voxel offset for TIFF inputs; ignored for STLs.
        data: The loaded ``uint32`` label volume (set during construction).
        voxel_offset: The volume's voxel offset (set during construction).
    """

    file_paths: Sequence[AnyPath]
    resolution: Sequence[int]
    min_chunks: int = 8
    voxel_offset_override: Optional[Sequence[int]] = None

    def __post_init__(self):
        self.file_paths = [os.fspath(path) for path in self.file_paths]
        self.resolution = tuple(self.resolution)
        if len(self.resolution) != 3:
            raise ValueError("resolution must contain exactly three values (X, Y, Z)")
        if (
            self.voxel_offset_override is not None
            and len(self.voxel_offset_override) != 3
        ):
            raise ValueError(
                "voxel_offset_override must contain exactly three values (X, Y, Z)"
            )

        exts = {os.path.splitext(path)[1].lower() for path in self.file_paths}
        if exts <= {".tif", ".tiff"}:
            if len(self.file_paths) != 1:
                raise ValueError("Only one TIFF file supported at a time")
            self.data = self._load_tiff(self.file_paths[0])
            override = self.voxel_offset_override or [0, 0, 0]
            self.voxel_offset = list(override)
        elif exts <= {".stl"}:
            self.data = self._load_stls()
        else:
            raise ValueError(f"Mixed or unsupported file types: {exts}")

    def _load_tiff(self, path: AnyPath) -> np.ndarray:
        """Read a TIFF into a uint32 ``(X, Y, Z)`` label volume."""
        data = tifffile.imread(path)
        logger.info(f"Loaded TIFF shape: {data.shape}, dtype: {data.dtype}")
        data = ensure_uint32_labels(data)
        if data.ndim == 3:
            data = np.transpose(data, (2, 1, 0))
        return data

    def _load_stls(self) -> np.ndarray:
        """Voxelize the STL files into one volume, one label ID per file."""
        origins = []
        grids = []
        for stl_file in self.file_paths:
            mesh = trimesh.load(os.fspath(stl_file), force="mesh")
            assert isinstance(mesh, trimesh.Trimesh)  # force="mesh" guarantees this
            logger.info(
                f"Loaded STL {os.path.basename(stl_file)}: "
                f"{len(mesh.vertices)} verts, {len(mesh.faces)} faces"
            )
            mesh.vertices /= np.array(self.resolution, dtype=float)
            voxel_grid = mesh.voxelized(pitch=1.0).fill()
            origin = np.round(voxel_grid.transform[:3, 3]).astype(int)
            origins.append(origin)
            grids.append(voxel_grid.matrix)

        global_min = np.min(origins, axis=0)
        global_max = np.max(
            [origin + np.array(grid.shape) for origin, grid in zip(origins, grids)],
            axis=0,
        )
        volume_shape = tuple(global_max - global_min)
        self.voxel_offset = global_min.tolist()

        data = np.zeros(volume_shape, dtype=np.uint32)
        for label_id, (origin, grid) in enumerate(zip(origins, grids), start=1):
            offset = origin - global_min
            slices = tuple(slice(o, o + size) for o, size in zip(offset, grid.shape))
            data[slices][grid] = np.uint32(label_id)

        logger.info(
            f"Combined {len(self.file_paths)} STL files into "
            f"volume {volume_shape}, labels 1-{len(self.file_paths)}"
        )
        return data

    def compute_chunk_size(self) -> Tuple[int, int, int]:
        return compute_chunk_size(self.data.shape, self.min_chunks)

    def compute_translation_nm(self) -> Tuple[int, int, int]:
        """Return the voxel offset converted to physical nanometers."""
        x, y, z = (
            offset * resolution
            for offset, resolution in zip(self.voxel_offset, self.resolution)
        )
        return (x, y, z)

    def build_info(self) -> dict:
        """Build the Neuroglancer precomputed ``info`` dict for this volume."""
        chunk_size = self.compute_chunk_size()
        translation_nm = self.compute_translation_nm()
        logger.info(
            f"Computed chunk_size={chunk_size}, voxel_offset={self.voxel_offset} "
            f"(= {translation_nm} nm)"
        )
        return {
            "data_type": "uint32",
            "num_channels": 1,
            "type": "segmentation",
            "scales": [
                {
                    "key": (
                        f"{self.resolution[0]}_{self.resolution[1]}_"
                        f"{self.resolution[2]}"
                    ),
                    "resolution": list(self.resolution),
                    "size": list(self.data.shape),
                    "voxel_offset": list(self.voxel_offset),
                    "chunk_sizes": [list(chunk_size)],
                    "encoding": "raw",
                }
            ],
        }


def _normalize_label_overrides(labels: Optional[LabelOverrides]) -> dict:
    """Coerce a mapping or ``ID:NAME`` strings into a ``{int: str}`` mapping."""
    if labels is None:
        return {}
    if isinstance(labels, Mapping):
        return {int(label_id): str(name) for label_id, name in labels.items()}
    if isinstance(labels, (str, bytes)):
        raise TypeError("labels must be a mapping or a sequence of ID:NAME strings")
    return parse_labels(labels)


def _find_input_files(input_path: str) -> list:
    """Return the TIFF/STL file(s) for a single path or a directory of them."""
    if os.path.isfile(input_path):
        return [input_path]
    candidates = sorted(
        filename
        for filename in os.listdir(input_path)
        if os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS
    )
    if not candidates:
        raise ValueError(f"No TIFF or STL files found in {input_path}")
    return [os.path.join(input_path, filename) for filename in candidates]


def _volume_name(input_path: str) -> str:
    """Derive the dataset name from the input file or directory."""
    if os.path.isfile(input_path):
        return os.path.splitext(os.path.basename(input_path))[0]
    return os.path.basename(os.path.normpath(input_path))


def _resolved_output_dir(input_path: str, output_base: Optional[str]) -> str:
    """Resolve the final ``<base>/<name>`` dataset output directory."""
    volume_name = _volume_name(input_path)
    if output_base:
        return os.path.join(output_base, volume_name)
    if os.path.isfile(input_path):
        return os.path.join(os.path.dirname(input_path), volume_name)
    return input_path


@dataclass
class MeshConverter:
    """Configurable TIFF/STL to Neuroglancer mesh conversion.

    Holds every conversion option as an attribute so a run can be configured
    once and executed with `run` (aliased as ``convert``). `tiff_to_mesh` is the
    thin functional wrapper around this class. ``output_dir`` is treated as a
    base directory: the input's name is appended to it and `run` returns that
    final dataset directory.

    Attributes:
        input_path: A ``.tif``/``.tiff``/``.stl`` file, or a directory
            containing exactly one such file.
        output_dir: Base output directory; the input name is appended. Defaults
            to the input's own directory when ``None``.
        resolution: Output voxel size in nm, as ``(X, Y, Z)``.
        voxel_offset: Voxel offset for TIFF inputs (see `DEFAULT_VOXEL_OFFSET`);
            ignored for STL inputs.
        unsharded: Emit the unsharded mesh format instead of the sharded default.
        labels: Segment-name overrides as ``{id: name}`` or ``"ID:NAME"``
            strings; highest priority.
        label_file: CSV of ``id,name`` segment names; overridden by ``labels``.
        setgit: Initialize a git repo in the output directory.
        push: GitHub repo name to create and push to; implies ``setgit`` and is
            mutually exclusive with it.
        mesh_dir: Name of the mesh subdirectory within the dataset.
    """

    input_path: AnyPath
    output_dir: Optional[AnyPath] = None
    resolution: Sequence[int] = DEFAULT_RESOLUTION
    voxel_offset: Sequence[int] = DEFAULT_VOXEL_OFFSET
    unsharded: bool = False
    labels: Optional[LabelOverrides] = None
    label_file: Optional[AnyPath] = None
    setgit: bool = False
    push: Optional[str] = None
    mesh_dir: str = MESH_DIR

    def run(self) -> str:
        """Run the full conversion and return the dataset directory.

        Loads the input into a label volume, writes the precomputed
        segmentation, meshes it (sharded by default, unsharded if configured),
        attaches segment properties, and optionally sets up or pushes a git
        repo. Existing mesh output is regenerated.

        Returns:
            Path to the dataset directory (``<output_dir>/<input-name>``).

        Raises:
            ValueError: If ``resolution`` or ``voxel_offset`` is not length 3,
                or if both ``setgit`` and ``push`` are set.
        """
        input_path = os.fspath(self.input_path)
        output_base = (
            os.fspath(self.output_dir) if self.output_dir is not None else None
        )
        resolution = tuple(self.resolution)
        voxel_offset = tuple(self.voxel_offset)
        if len(resolution) != 3:
            raise ValueError("resolution must contain exactly three values (X, Y, Z)")
        if len(voxel_offset) != 3:
            raise ValueError("voxel_offset must contain exactly three values (X, Y, Z)")
        if self.setgit and self.push:
            raise ValueError("setgit and push are mutually exclusive")

        if list(voxel_offset) == list(DEFAULT_VOXEL_OFFSET):
            logger.info(
                "Using default CRANTb voxel offset [-54, -54, -3]. "
                "Pass --voxel-offset 0 0 0 for non-CRANTb TIFFs."
            )

        input_files = _find_input_files(input_path)
        output_dir = _resolved_output_dir(input_path, output_base)
        mesh_dir = validate_mesh_dir(self.mesh_dir, output_dir)
        mesh_output_dir = os.path.join(output_dir, mesh_dir)
        cloudvolume_path = f"file://{output_dir}"

        if os.path.exists(mesh_output_dir):
            shutil.rmtree(mesh_output_dir)

        entry = MeshEntryLabels(
            file_paths=input_files,
            resolution=resolution,
            voxel_offset_override=voxel_offset,
        )
        data = entry.data
        info = entry.build_info()

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "info"), "w") as file:
            json.dump(info, file)

        volume = CloudVolume(cloudvolume_path, compress=False)
        volume[:] = data[:]
        logger.info(f"Volume shape: {volume.shape}, bounds: {volume.bounds}")

        task_queue = LocalTaskQueue(parallel=4)
        if self.unsharded:
            logger.info("Step 1: Creating unsharded mesh fragments...")
            mesh_tasks = tc.create_meshing_tasks(
                layer_path=cloudvolume_path,
                mip=0,
                shape=(256, 256, 256),
                simplification=True,
                max_simplification_error=40,
                mesh_dir=mesh_dir,
                sharded=False,
                spatial_index=False,
            )
            task_queue.insert(mesh_tasks)
            task_queue.execute()
            logger.info("Mesh fragments generated.")

            logger.info("Step 2: Creating unsharded multi-resolution draco meshes...")
            multires_tasks = tc.create_unsharded_multires_mesh_tasks(
                cloudpath=cloudvolume_path,
                num_lod=2,
                mesh_dir=mesh_dir,
                vertex_quantization_bits=16,
                min_chunk_size=(128, 128, 128),
            )
            task_queue.insert(multires_tasks)
            task_queue.execute()
            logger.info("Multi-resolution meshes generated.")

            import glob

            for filename in glob.glob(os.path.join(mesh_output_dir, "*:*")):
                os.remove(filename)
        else:
            logger.info("Step 1: Creating sharded mesh fragments...")
            mesh_tasks = tc.create_meshing_tasks(
                layer_path=cloudvolume_path,
                mip=0,
                shape=(256, 256, 256),
                simplification=True,
                max_simplification_error=40,
                mesh_dir=mesh_dir,
                sharded=True,
                spatial_index=True,
                compress="gzip",
            )
            task_queue.insert(mesh_tasks)
            task_queue.execute()
            logger.info("Mesh fragments generated.")

            logger.info("Step 2: Creating sharded multi-resolution draco meshes...")
            multires_tasks = tc.create_sharded_multires_mesh_tasks(
                cloudpath=cloudvolume_path,
                num_lod=2,
                mesh_dir=mesh_dir,
                vertex_quantization_bits=16,
                min_chunk_size=(128, 128, 128),
                draco_compression_level=7,
                shard_index_bytes=2**13,
                minishard_index_bytes=2**15,
            )
            task_queue.insert(multires_tasks)
            task_queue.execute()
            logger.info("Multi-resolution meshes generated.")

            import glob

            for filename in glob.glob(os.path.join(mesh_output_dir, "*.frags")):
                os.remove(filename)

        label_names = build_label_names_for_inputs(input_files, data)
        if self.label_file:
            label_names.update(parse_label_csv(os.fspath(self.label_file)))
        label_names.update(_normalize_label_overrides(self.labels))
        write_segment_properties(output_dir, label_names)

        with open(os.path.join(output_dir, "info")) as file:
            info_on_disk = json.load(file)
        info_on_disk["segment_properties"] = SEGMENT_PROPS_DIR
        with open(os.path.join(output_dir, "info"), "w") as file:
            json.dump(info_on_disk, file)

        volume = CloudVolume(cloudvolume_path)
        logger.info(f"Mesh info: {volume.mesh.meta.info}")

        if self.push:
            push_to_github(output_dir, self.push)
        elif self.setgit:
            git_dir = os.path.join(output_dir, ".git")
            if not os.path.exists(git_dir):
                subprocess.run(["git", "init"], cwd=output_dir, capture_output=True)
                logger.info(f"Git repo initialized in {output_dir}")

        logger.info("Done!")
        return output_dir

    convert = run


def tiff_to_mesh(
    input_path: AnyPath,
    output_dir: Optional[AnyPath] = None,
    *,
    resolution: Sequence[int] = DEFAULT_RESOLUTION,
    voxel_offset: Sequence[int] = DEFAULT_VOXEL_OFFSET,
    unsharded: bool = False,
    labels: Optional[LabelOverrides] = None,
    label_file: Optional[AnyPath] = None,
    setgit: bool = False,
    push: Optional[str] = None,
    mesh_dir: str = MESH_DIR,
) -> str:
    """Convert a TIFF/STL input to a Neuroglancer mesh dataset.

    One-call wrapper around `MeshConverter`; ``create_mesh`` is an exact alias.
    See `MeshConverter` for the object-oriented form.

    Args:
        input_path: A ``.tif``/``.tiff``/``.stl`` file, or a directory holding
            exactly one such file.
        output_dir: Base output directory; the input name is appended. Defaults
            to the input's own directory when ``None``.
        resolution: Output voxel size in nm, as ``(X, Y, Z)``.
        voxel_offset: Voxel offset for TIFF inputs (see `DEFAULT_VOXEL_OFFSET`);
            ignored for STL inputs, which derive it from geometry.
        unsharded: Emit the unsharded mesh format instead of the sharded default.
        labels: Segment-name overrides as ``{id: name}`` or ``"ID:NAME"``
            strings; takes priority over ``label_file`` and auto-derived names.
        label_file: CSV of ``id,name`` segment names.
        setgit: Initialize a git repo in the output directory.
        push: GitHub repo name to create and push to; implies ``setgit`` and is
            mutually exclusive with it.
        mesh_dir: Name of the mesh subdirectory within the dataset.

    Returns:
        Path to the dataset directory (``<output_dir>/<input-name>``).

    Raises:
        ValueError: If ``resolution`` or ``voxel_offset`` is not length 3, or if
            both ``setgit`` and ``push`` are set.
    """
    return MeshConverter(
        input_path=input_path,
        output_dir=output_dir,
        resolution=resolution,
        voxel_offset=voxel_offset,
        unsharded=unsharded,
        labels=labels,
        label_file=label_file,
        setgit=setgit,
        push=push,
        mesh_dir=mesh_dir,
    ).run()


create_mesh = tiff_to_mesh


def build_parser() -> argparse.ArgumentParser:
    """Build the ``tiff_to_mesh`` CLI argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--d", required=True, help="Directory or path to a TIFF/STL file"
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Base output directory; files are written to <out>/<input_name> "
            "(default: same as input directory)"
        ),
    )
    parser.add_argument(
        "--res",
        nargs=3,
        type=int,
        default=list(DEFAULT_RESOLUTION),
        metavar=("X", "Y", "Z"),
        help="Output resolution in nm for aligned meshes (default: 800 800 840)",
    )
    parser.add_argument(
        "--voxel-offset",
        nargs=3,
        type=int,
        default=list(DEFAULT_VOXEL_OFFSET),
        metavar=("X", "Y", "Z"),
        help=(
            "Voxel offset for TIFF inputs (default: -54 -54 -3 for CRANTb atlas "
            "alignment). STL inputs derive offset from geometry."
        ),
    )
    parser.add_argument(
        "--unsharded",
        action="store_true",
        help="Use unsharded format (default: sharded)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        metavar="ID:NAME",
        default=None,
        help=(
            "Manual segment names, e.g. --labels 1:ellipsoid_body "
            "2:fan_shaped_body (overrides auto-derived names)"
        ),
    )
    parser.add_argument(
        "--label-file",
        default=None,
        metavar="CSV",
        help=(
            "CSV file with id,name columns for segment names (overrides auto-derived "
            "names, --labels takes priority)"
        ),
    )
    git_group = parser.add_mutually_exclusive_group()
    git_group.add_argument(
        "--setgit", action="store_true", help="Initialize git repo in output directory"
    )
    git_group.add_argument(
        "--push",
        metavar="REPO_NAME",
        default=None,
        help=(
            "Create a public GitHub repo, push output, and print Neuroglancer raw "
            "link (implies --setgit)"
        ),
    )
    return parser


def main(argv=None) -> None:
    """CLI entry point: parse arguments and run the conversion."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    tiff_to_mesh(
        input_path=args.d,
        output_dir=args.out,
        resolution=args.res,
        voxel_offset=args.voxel_offset,
        unsharded=args.unsharded,
        labels=args.labels,
        label_file=args.label_file,
        setgit=args.setgit,
        push=args.push,
    )


__all__ = [
    "DEFAULT_RESOLUTION",
    "DEFAULT_VOXEL_OFFSET",
    "MeshConverter",
    "MeshEntryLabels",
    "build_parser",
    "create_mesh",
    "ensure_uint32_labels",
    "main",
    "tiff_to_mesh",
]
