import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
from itertools import combinations

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
    write_segment_properties,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def merge_datastacks(datastack_dirs, output_dir, mesh_dir, unsharded,
                     manual_labels=None, exclusions=None, source_properties=None):
    """Merge N precomputed datastacks into a standalone mesh dataset.

    Each datastack is meshed independently in its own padded volume so
    overlapping structures don't interfere with each other's mesh surfaces.
    The per-segment mesh files are then combined into one mesh-only output.

    Args:
        exclusions: {dirname: set_of_old_label_ids} — labels to skip.
        source_properties: {dirname: {label_id: name} | None} — from segment_properties.

    Returns a label_map dict: {source_path: {old_label: new_label}}.
    """
    sources = []
    for d in datastack_dirs:
        abspath = os.path.abspath(d)
        if not os.path.isdir(abspath):
            raise FileNotFoundError(f"Datastack directory not found: {abspath}")
        cv = CloudVolume(f"file://{abspath}", mip=0, fill_missing=True)
        sources.append((abspath, cv))

    output_dir = os.path.abspath(output_dir)
    source_paths = {abspath for abspath, _cv in sources}
    if output_dir in source_paths:
        raise ValueError("Output directory must be different from all input datastacks.")

    ref_res = sources[0][1].resolution.tolist()
    for abspath, cv in sources[1:]:
        if cv.resolution.tolist() != ref_res:
            raise ValueError(
                f"Resolution mismatch: {sources[0][0]} has {ref_res}, "
                f"but {abspath} has {cv.resolution.tolist()}. "
                f"All datastacks must share the same resolution to merge."
            )

    resolution = tuple(int(r) for r in ref_res)
    logger.info(f"Merging {len(sources)} datastacks at resolution {resolution}")

    volumes = []
    all_labels = []
    for abspath, cv in sources:
        data = np.squeeze(cv[:], axis=-1)  # (X, Y, Z, 1) -> (X, Y, Z)
        labels = sorted(set(np.unique(data).tolist()) - {0})
        if not labels:
            logger.warning(f"Datastack {abspath} has no non-zero labels, skipping.")
        logger.info(f"  {os.path.basename(abspath)}: shape={data.shape}, "
                     f"labels={labels}, offset={cv.bounds.minpt.tolist()}")
        volumes.append(data)
        all_labels.append(labels)

    label_map = {}
    next_id = 1
    total_excluded = 0
    for (abspath, _cv), labels in zip(sources, all_labels):
        ds_name = os.path.basename(abspath)
        ds_exclusions = exclusions.get(ds_name, set()) if exclusions else set()
        remap = {}
        skipped = []
        for old_label in labels:
            if int(old_label) in ds_exclusions:
                skipped.append(int(old_label))
                continue
            remap[int(old_label)] = next_id
            next_id += 1
        label_map[abspath] = remap
        if skipped:
            total_excluded += len(skipped)
            logger.info(f"  Excluded labels from {ds_name}: {skipped}")

    total_included = next_id - 1
    if total_included > UINT32_MAX:
        raise ValueError(f"Combined label count ({total_included}) exceeds uint32 max.")
    if total_excluded:
        logger.info(f"Label remapping: {total_included} included, {total_excluded} excluded, "
                     f"across {len(sources)} datastacks")
    else:
        logger.info(f"Label remapping: {total_included} total labels across {len(sources)} datastacks")
    # union bounding box
    all_minpts = np.array([cv.bounds.minpt for _, cv in sources])
    all_maxpts = np.array([cv.bounds.maxpt for _, cv in sources])
    union_min = np.min(all_minpts, axis=0)
    union_max = np.max(all_maxpts, axis=0)
    union_shape = tuple(int(x) for x in (union_max - union_min))
    voxel_offset = [int(x) for x in union_min]
    chunk_size = compute_chunk_size(union_shape)
    scale_key = f"{resolution[0]}_{resolution[1]}_{resolution[2]}"

    logger.info(f"Union bounding box: min={union_min.tolist()}, max={union_max.tolist()}, "
                f"shape={union_shape}")

    for (i, (path_a, cv_a)), (j, (path_b, cv_b)) in combinations(enumerate(sources), 2):
        overlap = Bbox.intersection(cv_a.bounds, cv_b.bounds)
        if not overlap.empty():
            logger.warning(
                f"Datastacks overlap: {os.path.basename(path_a)} and "
                f"{os.path.basename(path_b)}, overlap region {overlap} "
                f"({overlap.volume()} voxels). Meshes are generated independently "
                f"so both render correctly."
            )

    volume_info = {
        "data_type": "uint32",
        "mesh": mesh_dir,
        "num_channels": 1,
        "type": "segmentation",
        "scales": [{
            "key": scale_key,
            "resolution": list(resolution),
            "size": list(union_shape),
            "voxel_offset": voxel_offset,
            "chunk_sizes": [list(chunk_size)],
            "encoding": "raw",
        }]
    }

    temp_dirs = []
    aggregate_dir = None
    original_aggregate_dir = None
    try:
        for (abspath, cv), data, remap in zip(sources, volumes, label_map.values()):
            if not remap:
                continue

            ds_name = os.path.basename(abspath)
            logger.info(f"Processing {ds_name} ({len(remap)} labels) independently...")

            remapped = remap_labels_sparse(data, remap)

            padded = np.zeros(union_shape, dtype=np.uint32)
            offset = (cv.bounds.minpt - union_min).astype(int)
            slices = tuple(
                slice(int(o), int(o) + s)
                for o, s in zip(offset, data.shape)
            )
            padded[slices] = remapped

            temp_dir = tempfile.mkdtemp(prefix=f"merge_{ds_name}_")
            temp_dirs.append(temp_dir)

            with open(os.path.join(temp_dir, "info"), "w") as f:
                json.dump(volume_info, f)

            temp_cv_path = f"file://{temp_dir}"
            temp_cv = CloudVolume(temp_cv_path, compress=False)
            temp_cv[:] = padded[:]

            if unsharded:
                forge_unsharded_mesh_fragments(temp_cv_path, mesh_dir)
                finalize_unsharded_meshes(temp_cv_path, mesh_dir)
            else:
                forge_sharded_mesh_fragments(temp_cv_path, mesh_dir)
            logger.info(f"  Meshes generated for {ds_name}")

        if not temp_dirs:
            raise ValueError(
                "No labels remain after exclusions -- nothing to merge. "
                "Check your --exclude arguments."
            )

        if unsharded:
            aggregate_dir = tempfile.mkdtemp(prefix="merge_output_unsharded_")
            shutil.copy2(
                os.path.join(temp_dirs[0], mesh_dir, "info"),
                os.path.join(aggregate_dir, "info"),
            )
            for temp_dir in temp_dirs:
                temp_mesh = os.path.join(temp_dir, mesh_dir)
                for fname in os.listdir(temp_mesh):
                    if fname == "info":
                        continue
                    shutil.copy2(
                        os.path.join(temp_mesh, fname),
                        os.path.join(aggregate_dir, fname),
                    )
        else:
            aggregate_dir = tempfile.mkdtemp(prefix="merge_output_sharded_")
            with open(os.path.join(aggregate_dir, "info"), "w") as f:
                json.dump(volume_info, f)
            aggregate_mesh_dir = os.path.join(aggregate_dir, mesh_dir)
            os.makedirs(aggregate_mesh_dir, exist_ok=True)
            shutil.copy2(
                os.path.join(temp_dirs[0], mesh_dir, "info"),
                os.path.join(aggregate_mesh_dir, "info"),
            )
            for temp_dir in temp_dirs:
                temp_mesh = os.path.join(temp_dir, mesh_dir)
                for fname in os.listdir(temp_mesh):
                    if fname == "info":
                        continue
                    src = os.path.join(temp_mesh, fname)
                    dst = os.path.join(aggregate_mesh_dir, fname)
                    if os.path.isdir(src) and not os.path.islink(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
            finalize_sharded_meshes(f"file://{aggregate_dir}", mesh_dir)

            original_aggregate_dir = aggregate_dir
            sharded_publish_dir = tempfile.mkdtemp(prefix="merge_publish_sharded_")
            for fname in os.listdir(aggregate_mesh_dir):
                src = os.path.join(aggregate_mesh_dir, fname)
                dst = os.path.join(sharded_publish_dir, fname)
                if os.path.isdir(src) and not os.path.islink(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            aggregate_dir = sharded_publish_dir

        publish_mesh_files(aggregate_dir, output_dir)

    finally:
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
        if aggregate_dir:
            shutil.rmtree(aggregate_dir, ignore_errors=True)
        if original_aggregate_dir and original_aggregate_dir != aggregate_dir:
            shutil.rmtree(original_aggregate_dir, ignore_errors=True)

    label_names = {}
    for (abspath, _cv), remap in zip(sources, label_map.values()):
        ds_name = os.path.basename(abspath)
        props = source_properties.get(ds_name) if source_properties else None
        stl_files = sorted(
            f for f in os.listdir(abspath)
            if f.lower().endswith('.stl')
        )
        old_labels_sorted = sorted(remap.keys())
        for old_label in old_labels_sorted:
            new_label = remap[old_label]
            if props and old_label in props:
                name = props[old_label]
            elif (old_label - 1) < len(stl_files):
                name = os.path.splitext(stl_files[old_label - 1])[0]
            else:
                name = f"{ds_name}_label_{old_label}"
            label_names[new_label] = name

    if manual_labels:
        for (abspath, _cv), remap in zip(sources, label_map.values()):
            ds_name = os.path.basename(abspath)
            if ds_name in manual_labels:
                for old_id, name in manual_labels[ds_name].items():
                    if old_id in remap:
                        label_names[remap[old_id]] = name

    write_segment_properties(output_dir, label_names)
    attach_segment_properties_to_info(os.path.join(output_dir, "info"))
    with open(os.path.join(output_dir, "info")) as f:
        logger.info(f"Mesh info: {json.load(f)}")

    label_map_path = os.path.join(output_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    logger.info(f"Label mapping written to {label_map_path}")

    return label_map


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple Neuroglancer precomputed mesh datastacks into one."
    )
    parser.add_argument(
        'datastacks', nargs='+', metavar='DIR',
        help='Two or more datastack directories to merge'
    )
    parser.add_argument(
        '--out', required=True,
        help='Output directory for the merged standalone mesh dataset'
    )
    parser.add_argument('--unsharded', action='store_true', help='Use unsharded format (default: sharded)')
    parser.add_argument(
        '--labels', nargs='+', default=None,
        help='Manual segment names grouped by source dir, e.g. --labels pb_glomeruli_meshes 1:glom_L5 2:glom_L6 protocerebralbridge_mesh 1:bridge'
    )
    parser.add_argument(
        '--exclude', nargs='+', default=None,
        help='Exclude segments by ID or name, grouped by source dir basename. '
             'E.g.: --exclude datastack_A 1 3 datastack_B PB  '
             '(override = exclude old segment + include replacement from another stack)'
    )
    git_group = parser.add_mutually_exclusive_group()
    git_group.add_argument('--setgit', action='store_true', help='Initialize git repo in output directory')
    git_group.add_argument('--push', metavar='REPO_NAME', default=None,
                           help='Create a public GitHub repo, push output, and print Neuroglancer raw link')
    args = parser.parse_args()

    if len(args.datastacks) < 2:
        parser.error("At least 2 datastack directories are required.")

    output_dir = args.out
    manual_labels = parse_grouped_labels(args.labels) if args.labels else None

    source_properties = {}
    for d in args.datastacks:
        ds_name = os.path.basename(os.path.abspath(d))
        source_properties[ds_name] = read_segment_properties(os.path.abspath(d))

    exclusions = None
    if args.exclude:
        known_dirnames = {os.path.basename(os.path.abspath(d)) for d in args.datastacks}
        raw_exclusions = parse_grouped_exclusions(args.exclude, known_dirnames)
        exclusions = resolve_exclusions(raw_exclusions, source_properties)

    label_map = merge_datastacks(
        datastack_dirs=args.datastacks,
        output_dir=output_dir,
        mesh_dir=MESH_DIR,
        unsharded=args.unsharded,
        manual_labels=manual_labels,
        exclusions=exclusions,
        source_properties=source_properties,
    )

    total_labels = sum(len(m) for m in label_map.values())
    logger.info(f"Merged {len(args.datastacks)} datastacks, {total_labels} total labels")

    if args.push:
        push_to_github(output_dir, args.push, force=True)
    elif args.setgit:
        git_dir = os.path.join(output_dir, ".git")
        if not os.path.exists(git_dir):
            subprocess.run(["git", "init"], cwd=output_dir, capture_output=True)
            logger.info(f"Git repo initialized in {output_dir}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
