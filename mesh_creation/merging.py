"""Neuroglancer datastack merging and label replacement workflows."""

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox

from shared import (
    MESH_DIR,
    UINT32_MAX,
    attach_segment_properties_to_info,
    compute_chunk_size,
    finalize_sharded_meshes,
    finalize_unsharded_meshes,
    forge_sharded_mesh_fragments,
    forge_unsharded_mesh_fragments,
    parse_grouped_exclusions,
    parse_grouped_labels,
    publish_mesh_files,
    push_to_github,
    read_segment_properties,
    remap_labels_sparse,
    resolve_exclusions,
    validate_mesh_dir,
    write_segment_properties,
)

logger = logging.getLogger(__name__)

LabelSelector = Union[int, str]
GroupedSelectors = Mapping[os.PathLike, Iterable[LabelSelector]]
GroupedLabels = Mapping[os.PathLike, Mapping[int, str]]


def _source_paths(datastack_dirs: Sequence[os.PathLike]) -> list:
    if len(datastack_dirs) < 2:
        raise ValueError("At least 2 datastack directories are required.")

    paths = []
    resolved_paths = set()
    for directory in datastack_dirs:
        path = os.path.abspath(os.fspath(directory))
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Datastack directory not found: {path}")
        resolved = os.path.realpath(path)
        if resolved in resolved_paths:
            raise ValueError(f"Duplicate datastack directory: {path}")
        resolved_paths.add(resolved)
        paths.append(path)
    return paths


def _validate_output_path(output_dir: os.PathLike, source_paths: Sequence[str]) -> str:
    output_path = os.path.abspath(os.fspath(output_dir))
    output_resolved = os.path.realpath(output_path)
    for source_path in source_paths:
        source_resolved = os.path.realpath(source_path)
        try:
            common = os.path.commonpath([output_resolved, source_resolved])
        except ValueError:
            continue
        if common in {output_resolved, source_resolved}:
            raise ValueError(
                "Output directory must be separate from all input datastacks "
                "and may not contain or be contained by one."
            )
    return output_path


def _resolve_source_key(key: os.PathLike, source_paths: Sequence[str]) -> str:
    key_string = os.fspath(key)
    absolute_key = os.path.abspath(key_string)
    direct = [path for path in source_paths if path == absolute_key]
    if direct:
        return direct[0]

    real_key = os.path.realpath(absolute_key)
    resolved = [path for path in source_paths if os.path.realpath(path) == real_key]
    if resolved:
        return resolved[0]

    basename_matches = [
        path for path in source_paths if os.path.basename(path) == key_string
    ]
    if len(basename_matches) == 1:
        return basename_matches[0]
    if len(basename_matches) > 1:
        raise ValueError(
            f"Ambiguous datastack name '{key_string}'; use an absolute path instead."
        )
    raise ValueError(
        f"Unknown datastack '{key_string}'. Available sources: {source_paths}"
    )


def _normalize_grouped_mapping(
    grouped: Optional[Mapping], source_paths: Sequence[str]
) -> dict:
    if grouped is None:
        return {}
    normalized = {}
    for source_key, value in grouped.items():
        source_path = _resolve_source_key(source_key, source_paths)
        if source_path in normalized:
            raise ValueError(f"Datastack configured more than once: {source_path}")
        normalized[source_path] = value
    return normalized


def _load_source_properties(
    source_paths: Sequence[str], supplied: Optional[Mapping]
) -> dict:
    supplied_by_path = _normalize_grouped_mapping(supplied, source_paths)
    properties = {}
    for source_path in source_paths:
        if source_path in supplied_by_path:
            properties[source_path] = supplied_by_path[source_path]
        else:
            properties[source_path] = read_segment_properties(source_path)
    return properties


def _resolve_selectors(
    selectors: Iterable[LabelSelector],
    properties: Optional[Mapping[int, str]],
    source_path: str,
    *,
    reject_ambiguous_names: bool = False,
) -> set:
    if isinstance(selectors, (str, bytes)):
        selectors = [selectors]
    reverse = {}
    if properties:
        for label_id, name in properties.items():
            reverse.setdefault(name, []).append(int(label_id))

    resolved = set()
    for selector in selectors:
        if isinstance(selector, (int, np.integer)) and not isinstance(selector, bool):
            resolved.add(int(selector))
            continue
        if not isinstance(selector, str):
            raise TypeError(
                f"Label selectors must be integer IDs or names, got {selector!r}"
            )
        if properties is None:
            raise ValueError(
                f"Cannot resolve label name '{selector}' for "
                f"{os.path.basename(source_path)}: no segment_properties found. "
                "Use numeric label ID instead."
            )
        matches = reverse.get(selector, [])
        if not matches:
            raise ValueError(
                f"Label name '{selector}' not found in "
                f"{os.path.basename(source_path)}. "
                f"Available names: {sorted(reverse.keys())}"
            )
        if reject_ambiguous_names and len(matches) > 1:
            raise ValueError(
                f"Label name '{selector}' is ambiguous in "
                f"{os.path.basename(source_path)}; matching IDs: {sorted(matches)}. "
                "Use a numeric label ID instead."
            )
        resolved.update(matches)
    return resolved


def _normalize_selectors_by_source(
    grouped: Optional[GroupedSelectors],
    source_paths: Sequence[str],
    source_properties: Mapping[str, Optional[Mapping[int, str]]],
) -> dict:
    normalized = _normalize_grouped_mapping(grouped, source_paths)
    return {
        source_path: _resolve_selectors(
            selectors, source_properties[source_path], source_path
        )
        for source_path, selectors in normalized.items()
    }


def _normalize_manual_labels(
    grouped: Optional[GroupedLabels], source_paths: Sequence[str]
) -> dict:
    normalized = _normalize_grouped_mapping(grouped, source_paths)
    return {
        source_path: {int(label_id): str(name) for label_id, name in labels.items()}
        for source_path, labels in normalized.items()
    }


def _read_present_labels(source_path: str) -> set:
    volume = CloudVolume(f"file://{source_path}", mip=0, fill_missing=True)
    data = np.squeeze(volume[:], axis=-1)
    return {int(label) for label in np.unique(data).tolist() if int(label) != 0}


def _build_label_map(
    source_paths: Sequence[str],
    all_labels: Sequence[Sequence[int]],
    exclusions_by_path: Mapping[str, set],
    inclusions_by_path: Mapping[str, set],
):
    """Build the deterministic source-to-output label plan."""
    label_map = {}
    skipped_by_path = {}
    next_id = 1
    for source_path, present_labels in zip(source_paths, all_labels):
        source_exclusions = exclusions_by_path.get(source_path, set())
        source_inclusions = inclusions_by_path.get(source_path)
        remap = {}
        skipped = []
        for old_label in present_labels:
            label_id = int(old_label)
            if label_id in source_exclusions or (
                source_inclusions is not None and label_id not in source_inclusions
            ):
                skipped.append(label_id)
                continue
            remap[label_id] = next_id
            next_id += 1
        label_map[source_path] = remap
        skipped_by_path[source_path] = skipped
    return label_map, skipped_by_path


def _validate_label_selections(
    source_paths: Sequence[str],
    all_labels: Sequence[Sequence[int]],
    exclusions_by_path: Mapping[str, set],
    inclusions_by_path: Mapping[str, set],
    manual_labels_by_path: Mapping[str, Mapping[int, str]],
) -> None:
    present_by_path = {
        source_path: {int(label) for label in present_labels}
        for source_path, present_labels in zip(source_paths, all_labels)
    }
    selections = (
        ("exclude", exclusions_by_path),
        ("include", inclusions_by_path),
        ("labels", manual_labels_by_path),
    )
    for option_name, grouped in selections:
        for source_path, requested in grouped.items():
            requested_ids = {int(label) for label in requested}
            if option_name == "include" and not requested_ids:
                raise ValueError(
                    f"include selection for {source_path} must contain at least one label"
                )
            missing = requested_ids - present_by_path[source_path]
            if missing:
                raise ValueError(
                    f"Labels requested by {option_name} are not present in "
                    f"{source_path}: {sorted(missing)}"
                )


def _merge_datastacks(
    datastack_dirs: Sequence[os.PathLike],
    output_dir: os.PathLike,
    mesh_dir: str,
    unsharded: bool,
    manual_labels: Optional[GroupedLabels] = None,
    exclusions: Optional[GroupedSelectors] = None,
    source_properties: Optional[Mapping] = None,
    *,
    include: Optional[GroupedSelectors] = None,
    validate_labels: bool = True,
) -> dict:
    """Merge precomputed datastacks into one standalone mesh dataset.

    This is the shared implementation behind the public API and compatibility
    wrapper. Group keys may be absolute source paths or unique basenames.

    Sources are meshed independently, then assigned contiguous output IDs in
    source order and ascending input-label order. The return value and
    ``label_map.json`` map absolute source paths to ``old_id -> new_id`` maps.

    Unsharded output is the safe default for independent-source aggregation.
    Passing ``unsharded=False`` retains the legacy CLI's sharded workflow.
    """
    mesh_dir = validate_mesh_dir(mesh_dir)
    source_paths = _source_paths(datastack_dirs)
    output_path = _validate_output_path(output_dir, source_paths)
    properties_by_path = _load_source_properties(source_paths, source_properties)
    exclusions_by_path = _normalize_selectors_by_source(
        exclusions, source_paths, properties_by_path
    )
    inclusions_by_path = _normalize_selectors_by_source(
        include, source_paths, properties_by_path
    )
    manual_labels_by_path = _normalize_manual_labels(manual_labels, source_paths)

    sources = []
    for source_path in source_paths:
        volume = CloudVolume(f"file://{source_path}", mip=0, fill_missing=True)
        sources.append((source_path, volume))

    reference_resolution = sources[0][1].resolution.tolist()
    for source_path, volume in sources[1:]:
        if volume.resolution.tolist() != reference_resolution:
            raise ValueError(
                f"Resolution mismatch: {sources[0][0]} has {reference_resolution}, "
                f"but {source_path} has {volume.resolution.tolist()}. "
                "All datastacks must share the same resolution to merge."
            )

    resolution = tuple(int(value) for value in reference_resolution)
    logger.info(f"Merging {len(sources)} datastacks at resolution {resolution}")

    volumes = []
    all_labels = []
    for source_path, volume in sources:
        data = np.squeeze(volume[:], axis=-1)
        present_labels = sorted(set(np.unique(data).tolist()) - {0})
        if not present_labels:
            logger.warning(f"Datastack {source_path} has no non-zero labels, skipping.")
        logger.info(
            f"  {os.path.basename(source_path)}: shape={data.shape}, "
            f"labels={present_labels}, offset={volume.bounds.minpt.tolist()}"
        )
        volumes.append(data)
        all_labels.append(present_labels)

    if validate_labels:
        _validate_label_selections(
            source_paths,
            all_labels,
            exclusions_by_path,
            inclusions_by_path,
            manual_labels_by_path,
        )

    label_map, skipped_by_path = _build_label_map(
        source_paths, all_labels, exclusions_by_path, inclusions_by_path
    )
    total_excluded = 0
    for source_path in source_paths:
        skipped = skipped_by_path[source_path]
        if skipped:
            total_excluded += len(skipped)
            logger.info(
                f"  Excluded labels from {os.path.basename(source_path)}: {skipped}"
            )

    total_included = sum(len(remap) for remap in label_map.values())
    if total_included > UINT32_MAX:
        raise ValueError(f"Combined label count ({total_included}) exceeds uint32 max.")
    if total_excluded:
        logger.info(
            f"Label remapping: {total_included} included, {total_excluded} excluded, "
            f"across {len(sources)} datastacks"
        )
    else:
        logger.info(
            f"Label remapping: {total_included} total labels across "
            f"{len(sources)} datastacks"
        )

    all_minimum_points = np.array([volume.bounds.minpt for _, volume in sources])
    all_maximum_points = np.array([volume.bounds.maxpt for _, volume in sources])
    union_minimum = np.min(all_minimum_points, axis=0)
    union_maximum = np.max(all_maximum_points, axis=0)
    union_shape = tuple(int(value) for value in (union_maximum - union_minimum))
    voxel_offset = [int(value) for value in union_minimum]
    chunk_size = compute_chunk_size(union_shape)
    scale_key = f"{resolution[0]}_{resolution[1]}_{resolution[2]}"

    logger.info(
        f"Union bounding box: min={union_minimum.tolist()}, "
        f"max={union_maximum.tolist()}, shape={union_shape}"
    )

    for (_i, (path_a, volume_a)), (_j, (path_b, volume_b)) in combinations(
        enumerate(sources), 2
    ):
        overlap = Bbox.intersection(volume_a.bounds, volume_b.bounds)
        if not overlap.empty():
            logger.warning(
                f"Datastacks overlap: {os.path.basename(path_a)} and "
                f"{os.path.basename(path_b)}, overlap region {overlap} "
                f"({overlap.volume()} voxels). Meshes are generated independently "
                "so both render correctly."
            )

    volume_info = {
        "data_type": "uint32",
        "mesh": mesh_dir,
        "num_channels": 1,
        "type": "segmentation",
        "scales": [
            {
                "key": scale_key,
                "resolution": list(resolution),
                "size": list(union_shape),
                "voxel_offset": voxel_offset,
                "chunk_sizes": [list(chunk_size)],
                "encoding": "raw",
            }
        ],
    }

    temporary_directories = []
    aggregate_dir = None
    original_aggregate_dir = None
    try:
        for (source_path, volume), data, remap in zip(
            sources, volumes, label_map.values()
        ):
            if not remap:
                continue

            source_name = os.path.basename(source_path)
            logger.info(
                f"Processing {source_name} ({len(remap)} labels) independently..."
            )
            remapped = remap_labels_sparse(data, remap)

            padded = np.zeros(union_shape, dtype=np.uint32)
            offset = (volume.bounds.minpt - union_minimum).astype(int)
            slices = tuple(
                slice(int(axis_offset), int(axis_offset) + size)
                for axis_offset, size in zip(offset, data.shape)
            )
            padded[slices] = remapped

            temporary_directory = tempfile.mkdtemp(prefix=f"merge_{source_name}_")
            temporary_directories.append(temporary_directory)
            with open(os.path.join(temporary_directory, "info"), "w") as file:
                json.dump(volume_info, file)

            temporary_cloud_path = f"file://{temporary_directory}"
            temporary_volume = CloudVolume(temporary_cloud_path, compress=False)
            temporary_volume[:] = padded[:]

            if unsharded:
                forge_unsharded_mesh_fragments(temporary_cloud_path, mesh_dir)
                finalize_unsharded_meshes(temporary_cloud_path, mesh_dir)
            else:
                forge_sharded_mesh_fragments(temporary_cloud_path, mesh_dir)
            logger.info(f"  Meshes generated for {source_name}")

        if not temporary_directories:
            raise ValueError(
                "No labels remain after exclusions -- nothing to merge. "
                "Check your --exclude arguments."
            )

        if unsharded:
            aggregate_dir = tempfile.mkdtemp(prefix="merge_output_unsharded_")
            shutil.copy2(
                os.path.join(temporary_directories[0], mesh_dir, "info"),
                os.path.join(aggregate_dir, "info"),
            )
            for temporary_directory in temporary_directories:
                temporary_mesh = os.path.join(temporary_directory, mesh_dir)
                for filename in os.listdir(temporary_mesh):
                    if filename == "info":
                        continue
                    shutil.copy2(
                        os.path.join(temporary_mesh, filename),
                        os.path.join(aggregate_dir, filename),
                    )
        else:
            aggregate_dir = tempfile.mkdtemp(prefix="merge_output_sharded_")
            with open(os.path.join(aggregate_dir, "info"), "w") as file:
                json.dump(volume_info, file)
            aggregate_mesh_dir = os.path.join(aggregate_dir, mesh_dir)
            os.makedirs(aggregate_mesh_dir, exist_ok=True)
            shutil.copy2(
                os.path.join(temporary_directories[0], mesh_dir, "info"),
                os.path.join(aggregate_mesh_dir, "info"),
            )
            for temporary_directory in temporary_directories:
                temporary_mesh = os.path.join(temporary_directory, mesh_dir)
                for filename in os.listdir(temporary_mesh):
                    if filename == "info":
                        continue
                    source = os.path.join(temporary_mesh, filename)
                    destination = os.path.join(aggregate_mesh_dir, filename)
                    if os.path.isdir(source) and not os.path.islink(source):
                        shutil.copytree(source, destination, dirs_exist_ok=True)
                    else:
                        shutil.copy2(source, destination)
            finalize_sharded_meshes(f"file://{aggregate_dir}", mesh_dir)

            original_aggregate_dir = aggregate_dir
            sharded_publish_dir = tempfile.mkdtemp(prefix="merge_publish_sharded_")
            for filename in os.listdir(aggregate_mesh_dir):
                source = os.path.join(aggregate_mesh_dir, filename)
                destination = os.path.join(sharded_publish_dir, filename)
                if os.path.isdir(source) and not os.path.islink(source):
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)
            aggregate_dir = sharded_publish_dir

        publish_mesh_files(aggregate_dir, output_path)

    finally:
        for temporary_directory in temporary_directories:
            shutil.rmtree(temporary_directory, ignore_errors=True)
        if aggregate_dir:
            shutil.rmtree(aggregate_dir, ignore_errors=True)
        if original_aggregate_dir and original_aggregate_dir != aggregate_dir:
            shutil.rmtree(original_aggregate_dir, ignore_errors=True)

    label_names = {}
    for (source_path, _volume), remap in zip(sources, label_map.values()):
        source_name = os.path.basename(source_path)
        properties = properties_by_path[source_path]
        stl_files = sorted(
            filename
            for filename in os.listdir(source_path)
            if filename.lower().endswith(".stl")
        )
        for old_label in sorted(remap.keys()):
            new_label = remap[old_label]
            if properties and old_label in properties:
                name = properties[old_label]
            elif (old_label - 1) < len(stl_files):
                name = os.path.splitext(stl_files[old_label - 1])[0]
            else:
                name = f"{source_name}_label_{old_label}"
            label_names[new_label] = name

    for source_path, label_overrides in manual_labels_by_path.items():
        remap = label_map[source_path]
        for old_label, name in label_overrides.items():
            if old_label in remap:
                label_names[remap[old_label]] = name

    write_segment_properties(output_path, label_names)
    attach_segment_properties_to_info(os.path.join(output_path, "info"))
    with open(os.path.join(output_path, "info")) as file:
        logger.info(f"Mesh info: {json.load(file)}")

    label_map_path = os.path.join(output_path, "label_map.json")
    with open(label_map_path, "w") as file:
        json.dump(label_map, file, indent=2)
    logger.info(f"Label mapping written to {label_map_path}")
    return label_map


def merge_datastacks(
    datastack_dirs: Sequence[os.PathLike],
    output_dir: os.PathLike,
    *,
    unsharded: bool = True,
    mesh_dir: str = MESH_DIR,
    labels: Optional[GroupedLabels] = None,
    exclude: Optional[GroupedSelectors] = None,
    include: Optional[GroupedSelectors] = None,
    source_properties: Optional[Mapping] = None,
    validate_labels: bool = True,
) -> dict:
    """Merge precomputed datastacks into one standalone mesh dataset.

    ``labels``, ``exclude``, and ``include`` are grouped by source. A group key
    may be an absolute source path or a basename that is unique among the
    inputs. Label selectors may be numeric IDs or names from segment properties.

    Sources are meshed independently, then assigned contiguous output IDs in
    source order and ascending input-label order. The return value and
    ``label_map.json`` map absolute source paths to ``old_id -> new_id`` maps.

    Unsharded output is the safe default for independent-source aggregation.
    Passing ``unsharded=False`` explicitly retains the legacy sharded workflow.
    By default, configured label IDs are checked against the source volumes;
    pass ``validate_labels=False`` only for legacy permissive behavior.
    """
    return _merge_datastacks(
        datastack_dirs=datastack_dirs,
        output_dir=output_dir,
        mesh_dir=mesh_dir,
        unsharded=unsharded,
        manual_labels=labels,
        exclusions=exclude,
        source_properties=source_properties,
        include=include,
        validate_labels=validate_labels,
    )


def replace_labels(
    base: os.PathLike,
    replacement: os.PathLike,
    labels: Iterable[LabelSelector],
    output_dir: os.PathLike,
    replacement_labels: Optional[Iterable[LabelSelector]] = None,
    *,
    mesh_dir: str = MESH_DIR,
    unsharded: bool = True,
    label_names: Optional[GroupedLabels] = None,
    exclude: Optional[GroupedSelectors] = None,
    source_properties: Optional[Mapping] = None,
) -> dict:
    """Replace a group of labels from one stack with geometry from another.

    This is the explicit library form of the existing CLI workaround:
    selected labels are excluded from ``base`` and the replacement
    stack is merged independently. By default every non-zero label from the
    replacement stack is included. ``replacement_labels`` can restrict that
    contribution. Output IDs follow normal deterministic merge remapping; this
    operation is not a voxelwise overwrite and does not preserve source IDs.
    """
    source_paths = _source_paths([base, replacement])
    _validate_output_path(output_dir, source_paths)
    properties_by_path = _load_source_properties(source_paths, source_properties)
    base_path, replacement_path = source_paths
    exclusions_by_path = _normalize_selectors_by_source(
        exclude, source_paths, properties_by_path
    )

    selected_base = _resolve_selectors(
        labels,
        properties_by_path[base_path],
        base_path,
        reject_ambiguous_names=True,
    )
    if not selected_base:
        raise ValueError("At least one base label must be selected for replacement.")
    exclusions_by_path.setdefault(base_path, set()).update(selected_base)
    present_base = _read_present_labels(base_path)
    missing_base = selected_base - present_base
    if missing_base:
        raise ValueError(
            f"Labels not present in base datastack {base_path}: {sorted(missing_base)}"
        )

    present_replacement = _read_present_labels(replacement_path)
    if not present_replacement:
        raise ValueError(
            f"Replacement datastack has no non-zero labels: {replacement_path}"
        )

    include = None
    selected_replacement = present_replacement
    if replacement_labels is not None:
        selected_replacement = _resolve_selectors(
            replacement_labels,
            properties_by_path[replacement_path],
            replacement_path,
            reject_ambiguous_names=True,
        )
        if not selected_replacement:
            raise ValueError("At least one replacement label must be selected.")
        missing_replacement = selected_replacement - present_replacement
        if missing_replacement:
            raise ValueError(
                f"Labels not present in replacement datastack {replacement_path}: "
                f"{sorted(missing_replacement)}"
            )
        include = {replacement_path: selected_replacement}

    replacement_exclusions = exclusions_by_path.get(replacement_path, set())
    if not (selected_replacement - replacement_exclusions):
        raise ValueError(
            "No replacement labels remain after applying the replacement "
            "selection and exclusions."
        )

    return merge_datastacks(
        [base_path, replacement_path],
        output_dir,
        mesh_dir=mesh_dir,
        unsharded=unsharded,
        labels=label_names,
        exclude=exclusions_by_path,
        source_properties=properties_by_path,
        include=include,
    )


replace_datastack_labels = replace_labels


@dataclass
class DatastackMerger:
    """Object-oriented configuration for merge and replacement workflows."""

    datastacks: Sequence[os.PathLike]
    output_dir: os.PathLike
    unsharded: bool = True
    mesh_dir: str = MESH_DIR
    label_names: Optional[GroupedLabels] = None
    exclude: Optional[GroupedSelectors] = None
    source_properties: Optional[Mapping] = None

    def run(self) -> dict:
        return merge_datastacks(
            self.datastacks,
            self.output_dir,
            mesh_dir=self.mesh_dir,
            unsharded=self.unsharded,
            labels=self.label_names,
            exclude=self.exclude,
            source_properties=self.source_properties,
        )

    merge = run

    def replace(
        self,
        labels: Iterable[LabelSelector],
        *,
        replacement_labels: Optional[Iterable[LabelSelector]] = None,
    ) -> dict:
        if len(self.datastacks) != 2:
            raise ValueError(
                "DatastackMerger.replace() requires exactly two datastacks: "
                "base first, replacement second."
            )
        return replace_labels(
            self.datastacks[0],
            self.datastacks[1],
            labels,
            self.output_dir,
            replacement_labels=replacement_labels,
            mesh_dir=self.mesh_dir,
            unsharded=self.unsharded,
            label_names=self.label_names,
            exclude=self.exclude,
            source_properties=self.source_properties,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple Neuroglancer precomputed mesh datastacks into one."
        )
    )
    parser.add_argument(
        "datastacks",
        nargs="+",
        metavar="DIR",
        help="Two or more datastack directories to merge",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for the merged standalone mesh dataset",
    )
    parser.add_argument(
        "--unsharded",
        action="store_true",
        help="Use unsharded format (default: sharded)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help=(
            "Manual segment names grouped by source dir, e.g. --labels "
            "pb_glomeruli_meshes 1:glom_L5 2:glom_L6 "
            "protocerebralbridge_mesh 1:bridge"
        ),
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        help=(
            "Exclude segments by ID or name, grouped by source dir basename. "
            "E.g.: --exclude datastack_A 1 3 datastack_B PB  "
            "(override = exclude old segment + include replacement from another stack)"
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
            "Create a public GitHub repo, push output, and print Neuroglancer raw link"
        ),
    )
    return parser


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.datastacks) < 2:
        parser.error("At least 2 datastack directories are required.")

    manual_labels = parse_grouped_labels(args.labels) if args.labels else None
    source_properties = {}
    for directory in args.datastacks:
        source_path = os.path.abspath(directory)
        source_properties[source_path] = read_segment_properties(source_path)

    exclusions = None
    if args.exclude:
        known_source_names = {
            os.path.basename(os.path.abspath(directory))
            for directory in args.datastacks
        }
        raw_exclusions = parse_grouped_exclusions(args.exclude, known_source_names)
        properties_by_name = {}
        for source_path, properties in source_properties.items():
            source_name = os.path.basename(source_path)
            properties_by_name[source_name] = properties
        exclusions = resolve_exclusions(raw_exclusions, properties_by_name)

    def expand_cli_groups(grouped):
        if grouped is None:
            return None
        return {
            source_path: grouped[source_name]
            for source_path in source_properties
            for source_name in [os.path.basename(source_path)]
            if source_name in grouped
        }

    label_map = merge_datastacks(
        datastack_dirs=args.datastacks,
        output_dir=args.out,
        mesh_dir=MESH_DIR,
        unsharded=args.unsharded,
        labels=expand_cli_groups(manual_labels),
        exclude=expand_cli_groups(exclusions),
        source_properties=source_properties,
        validate_labels=False,
    )
    total_labels = sum(len(mapping) for mapping in label_map.values())
    logger.info(
        f"Merged {len(args.datastacks)} datastacks, {total_labels} total labels"
    )

    if args.push:
        push_to_github(args.out, args.push, force=True)
    elif args.setgit:
        git_dir = os.path.join(args.out, ".git")
        if not os.path.exists(git_dir):
            subprocess.run(["git", "init"], cwd=args.out, capture_output=True)
            logger.info(f"Git repo initialized in {args.out}")
    logger.info("Done!")


__all__ = [
    "DatastackMerger",
    "build_parser",
    "main",
    "merge_datastacks",
    "replace_datastack_labels",
    "replace_labels",
]
