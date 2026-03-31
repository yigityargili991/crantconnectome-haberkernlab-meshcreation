import glob
import io
import json
import logging
import os
import re
import subprocess
import shutil

import numpy as np

logger = logging.getLogger(__name__)

UINT32_MAX = np.iinfo(np.uint32).max
MESH_DIR = "mesh"
_CHUNK_CANDIDATES = (32, 64, 128)


def get_github_username() -> str:
    result = subprocess.run(
        ["gh", "api", "user", "--jq", ".login"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to detect GitHub username via `gh` CLI. "
            f"Make sure `gh` is installed and authenticated.\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def push_to_github(output_dir: str, repo_name: str, *, force: bool = False):
    username = get_github_username()
    full_repo = f"{username}/{repo_name}"
    logger.info(f"Pushing to GitHub repo: {full_repo}")

    subprocess.run(
        ["git", "init", "-b", "main"], cwd=output_dir,
        capture_output=True, text=True, check=True
    )

    create_result = subprocess.run(
        ["gh", "repo", "create", full_repo, "--public"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL
    )
    if create_result.returncode != 0:
        if "already exists" in create_result.stderr:
            logger.warning(f"Repo {full_repo} already exists, pushing new commit to it.")
        else:
            raise RuntimeError(
                f"Failed to create GitHub repo: {create_result.stderr.strip()}"
            )

    subprocess.run(
        ["git", "add", "."], cwd=output_dir,
        capture_output=True, text=True, check=True
    )
    commit_result = subprocess.run(
        ["git", "commit", "-m", "new mesh"], cwd=output_dir,
        capture_output=True, text=True
    )
    if commit_result.returncode != 0:
        if "nothing to commit" in commit_result.stdout:
            logger.info("No new changes to commit, pushing existing HEAD.")
        else:
            raise RuntimeError(
                f"git commit failed: {commit_result.stderr.strip() or commit_result.stdout.strip()}"
            )

    remote_url = f"https://github.com/{full_repo}.git"
    subprocess.run(
        ["git", "remote", "remove", "origin"], cwd=output_dir,
        capture_output=True
    )
    subprocess.run(
        ["git", "remote", "add", "origin", remote_url], cwd=output_dir,
        capture_output=True, text=True, check=True
    )

    push_cmd = ["git", "push", "-u", "origin", "main"]
    if force:
        push_cmd.insert(2, "--force")
    push_result = subprocess.run(
        push_cmd, cwd=output_dir, capture_output=True, text=True
    )
    if push_result.returncode != 0:
        raise RuntimeError(
            f"Failed to push to GitHub:\n{push_result.stderr.strip()}"
        )

    hash_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=output_dir,
        capture_output=True, text=True, check=True
    )
    commit_hash = hash_result.stdout.strip()

    raw_link = (
        f"https://raw.githubusercontent.com/{full_repo}/"
        f"{commit_hash}|neuroglancer-precomputed:"
    )
    logger.info(f"Neuroglancer source URL:\n{raw_link}")


SEGMENT_PROPS_DIR = "segment_properties"


def parse_labels(labels_arg):
    """Parse a list of 'ID:NAME' strings into a {int: str} dict.

    Used by tiff_to_mesh where there's a single source.
    """
    mapping = {}
    for item in labels_arg:
        if ':' not in item:
            raise ValueError(f"Invalid label format '{item}', expected ID:NAME (e.g. 1:ellipsoid_body)")
        id_str, name = item.split(':', 1)
        mapping[int(id_str)] = name
    return mapping


def parse_label_csv(csv_path):
    """Parse a CSV file with id,name columns into a {int: str} dict.

    Spreadsheet-exported CSVs are tolerated: UTF-8 BOMs, common spreadsheet
    encodings, alternate delimiters, and reordered header columns.
    Blank lines and lines starting with '#' are ignored.
    """
    import csv

    def read_text_with_fallbacks(path):
        for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
            try:
                with open(path, newline="", encoding=encoding) as f:
                    return f.read()
            except UnicodeError:
                continue
        # latin-1 maps every byte 0x00-0xFF, so this is unreachable.
        raise RuntimeError(f"All encoding fallbacks exhausted for {path}")

    def sniff_dialect(text):
        candidate_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            candidate_lines.append(line)
            if len(candidate_lines) >= 5:
                break
        sample = "\n".join(candidate_lines)
        if sample:
            try:
                return csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error:
                pass

        class DefaultDialect(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            doublequote = True
            skipinitialspace = True
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL

        return DefaultDialect

    def normalize_header(value):
        return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")

    def detect_columns(row):
        normalized = [normalize_header(cell) for cell in row]
        id_aliases = {"id", "label_id", "segment_id"}
        name_aliases = {"name", "label", "label_name", "segment_name"}
        try:
            id_idx = next(i for i, cell in enumerate(normalized) if cell in id_aliases)
            name_idx = next(i for i, cell in enumerate(normalized) if cell in name_aliases)
        except StopIteration:
            return None
        return id_idx, name_idx

    def is_separator_preamble(row):
        return bool(row) and row[0].lower().startswith("sep=")

    mapping = {}
    text = read_text_with_fallbacks(csv_path)
    reader = csv.reader(io.StringIO(text), dialect=sniff_dialect(text))
    column_indices = None
    for row in reader:
        stripped_row = [cell.strip() for cell in row]
        if not stripped_row or not any(stripped_row):
            continue
        if stripped_row[0].startswith("#"):
            continue
        if column_indices is None and is_separator_preamble(stripped_row):
            continue
        if column_indices is None:
            detected = detect_columns(stripped_row)
            if detected is not None:
                column_indices = detected
                continue
            column_indices = (0, 1)

        id_idx, name_idx = column_indices
        if len(stripped_row) <= max(id_idx, name_idx):
            raise ValueError(f"Row missing required columns: {row}")
        try:
            label_id = int(stripped_row[id_idx])
        except ValueError:
            continue
        name = stripped_row[name_idx]
        if name == "":
            raise ValueError(f"Row missing name column: {row}")
        mapping[label_id] = name

    if not mapping and text.strip():
        logger.warning(
            "No labels parsed from %s. If your CSV has name,id column order, "
            "add a header row like 'name,id' or 'id,name'.",
            csv_path,
        )
    return mapping


def build_label_names_for_inputs(input_files, data):
    """Build default label names for TIFF/STL inputs.

    STL inputs use filenames as names. TIFF inputs preserve the underlying
    label IDs and assign a deterministic fallback name for each non-zero label.
    """
    if not input_files:
        return {}
    label_names = {}
    exts = {os.path.splitext(f)[1].lower() for f in input_files}
    if exts <= {'.stl'}:
        for i, fpath in enumerate(input_files, start=1):
            label_names[i] = os.path.splitext(os.path.basename(fpath))[0]
    else:
        volume_name = os.path.splitext(os.path.basename(input_files[0]))[0]
        unique_labels = sorted(set(np.unique(data).tolist()) - {0})
        for lid in unique_labels:
            label_names[int(lid)] = f"{volume_name}_label_{int(lid)}"
    return label_names


def parse_grouped_labels(labels_arg):
    """Parse labels grouped by source directory name.

    Format: DIR1 ID:NAME ID:NAME DIR2 ID:NAME ...
    Tokens without ':' are treated as directory names (section headers).
    Returns {dirname: {old_label_id: name}}.
    """
    groups = {}
    current_dir = None
    for item in labels_arg:
        if ':' not in item:
            current_dir = item
            groups[current_dir] = {}
        else:
            if current_dir is None:
                raise ValueError(
                    f"Got label '{item}' before any directory name. "
                    f"Format: DIRNAME 1:name1 2:name2 DIRNAME2 1:name3"
                )
            id_str, name = item.split(':', 1)
            groups[current_dir][int(id_str)] = name
    return groups


def parse_grouped_exclusions(exclude_arg, known_dirnames):
    """Parse --exclude tokens into {dirname: [int_or_str, ...]}.

    Tokens matching a known datastack dirname start a new group.
    Other tokens are label specifiers: tried as int (label ID) first,
    otherwise kept as str (label name).
    """
    groups = {}
    current_dir = None
    for token in exclude_arg:
        if token in known_dirnames:
            current_dir = token
            groups.setdefault(current_dir, [])
        else:
            if current_dir is None:
                raise ValueError(
                    f"Exclusion '{token}' appears before any directory name. "
                    f"Format: DIRNAME 1 3 DIRNAME2 PB  "
                    f"(known dirs: {sorted(known_dirnames)})"
                )
            try:
                groups[current_dir].append(int(token))
            except ValueError:
                groups[current_dir].append(token)
    return groups


def resolve_exclusions(exclusions, source_properties):
    """Convert name-based exclusions to label IDs.

    Args:
        exclusions: {dirname: [int_or_str, ...]} from parse_grouped_exclusions.
        source_properties: {dirname: {label_id: name} | None} from read_segment_properties.

    Returns: {dirname: set_of_old_label_ids}.
    """
    resolved = {}
    for dirname, specifiers in exclusions.items():
        props = source_properties.get(dirname)
        reverse = {}
        if props:
            for lid, name in props.items():
                reverse.setdefault(name, []).append(lid)
        ids = set()
        for spec in specifiers:
            if isinstance(spec, int):
                ids.add(spec)
            else:
                if props is None:
                    raise ValueError(
                        f"Cannot resolve label name '{spec}' for {dirname}: "
                        f"no segment_properties found. Use numeric label ID instead."
                    )
                if spec not in reverse:
                    raise ValueError(
                        f"Label name '{spec}' not found in {dirname}. "
                        f"Available names: {sorted(reverse.keys())}"
                    )
                ids.update(reverse[spec])
        resolved[dirname] = ids
    return resolved


def write_segment_properties(output_dir, label_names):
    """Write Neuroglancer segment_properties for named labels.

    Args:
        output_dir: Root datastack directory.
        label_names: dict mapping int label_id -> str name.
    """
    props_dir = os.path.join(output_dir, SEGMENT_PROPS_DIR)
    os.makedirs(props_dir, exist_ok=True)

    ids = [str(lid) for lid in sorted(label_names.keys())]
    values = [label_names[int(lid)] for lid in ids]

    props_info = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": ids,
            "properties": [
                {
                    "id": "label",
                    "type": "label",
                    "values": values,
                }
            ],
        },
    }

    with open(os.path.join(props_dir, "info"), "w") as f:
        json.dump(props_info, f, indent=2)
    logger.info(f"Segment properties written for {len(ids)} labels")


def read_segment_properties(datastack_dir):
    """Read segment_properties/info from a precomputed datastack.

    Returns {int_label_id: str_name} or None if not found/malformed.
    """
    props_path = os.path.join(datastack_dir, SEGMENT_PROPS_DIR, "info")
    if not os.path.isfile(props_path):
        return None
    try:
        with open(props_path) as f:
            props = json.load(f)
        inline = props["inline"]
        ids = inline["ids"]
        for prop in inline["properties"]:
            if prop.get("id") == "label" and prop.get("type") == "label":
                values = prop["values"]
                return {int(lid): name for lid, name in zip(ids, values)}
        logger.warning(f"No 'label' property found in {props_path}")
        return None
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Could not parse segment properties at {props_path}: {e}")
        return None


def compute_chunk_size(shape, min_chunks=8):
    """Pick the largest power-of-2 chunk per axis such that
    each axis has at least `min_chunks` chunks."""
    chunks = []
    for dim_size in shape:
        chunk = _CHUNK_CANDIDATES[0]
        for candidate in _CHUNK_CANDIDATES:
            if dim_size >= candidate * min_chunks:
                chunk = candidate
        chunks.append(chunk)
    return tuple(chunks)


def remap_labels_sparse(data, remap):
    """Remap a labeled volume without allocating up to max(label_id)."""
    unique_labels, inverse = np.unique(data, return_inverse=True)
    mapped = np.zeros(unique_labels.shape, dtype=np.uint32)
    for idx, old_label in enumerate(unique_labels.tolist()):
        mapped[idx] = remap.get(int(old_label), 0)
    return mapped[inverse].reshape(data.shape)


def prepare_clean_dir(path):
    """Remove all existing contents of a directory while preserving the directory."""
    os.makedirs(path, exist_ok=True)
    existing = [n for n in os.listdir(path) if n != ".git"]
    if existing:
        logger.warning(
            "Output directory '%s' is non-empty — the following will be removed: %s",
            path,
            ", ".join(sorted(existing)),
        )
    for name in existing:
        target = os.path.join(path, name)
        if os.path.isdir(target) and not os.path.islink(target):
            shutil.rmtree(target)
        else:
            os.remove(target)


def publish_mesh_files(mesh_src_dir, output_dir):
    """Publish a mesh directory as a standalone mesh dataset root."""
    prepare_clean_dir(output_dir)
    for name in os.listdir(mesh_src_dir):
        src = os.path.join(mesh_src_dir, name)
        dst = os.path.join(output_dir, name)
        if os.path.isdir(src) and not os.path.islink(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def attach_segment_properties_to_info(info_path, segment_properties_dir=SEGMENT_PROPS_DIR):
    with open(info_path) as f:
        info = json.load(f)
    info["segment_properties"] = segment_properties_dir
    with open(info_path, "w") as f:
        json.dump(info, f)


def _load_meshing_modules():
    from taskqueue import LocalTaskQueue
    import igneous.task_creation as tc
    return LocalTaskQueue, tc


def _execute_tasks(tasks):
    LocalTaskQueue, _tc = _load_meshing_modules()
    tq = LocalTaskQueue(parallel=4)
    tq.insert(tasks)
    tq.execute()


def forge_unsharded_mesh_fragments(cloudvolume_path, mesh_dir):
    """Run the first-pass unsharded meshing stage."""
    _LocalTaskQueue, tc = _load_meshing_modules()
    mesh_output_dir = os.path.join(
        cloudvolume_path.replace("file://", ""), mesh_dir
    )
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
    _execute_tasks(mesh_tasks)
    logger.info("Mesh fragments generated.")
    return mesh_output_dir


def finalize_unsharded_meshes(cloudvolume_path, mesh_dir):
    """Run the second-pass unsharded multi-resolution mesh stage."""
    _LocalTaskQueue, tc = _load_meshing_modules()
    mesh_output_dir = os.path.join(
        cloudvolume_path.replace("file://", ""), mesh_dir
    )
    logger.info("Step 2: Creating unsharded multi-resolution draco meshes...")
    multires_tasks = tc.create_unsharded_multires_mesh_tasks(
        cloudpath=cloudvolume_path,
        num_lod=2,
        mesh_dir=mesh_dir,
        vertex_quantization_bits=16,
        min_chunk_size=(128, 128, 128),
    )
    _execute_tasks(multires_tasks)
    logger.info("Multi-resolution meshes generated.")

    for f in glob.glob(os.path.join(mesh_output_dir, "*:*")):
        os.remove(f)
    return mesh_output_dir


def forge_sharded_mesh_fragments(cloudvolume_path, mesh_dir):
    """Run the first-pass sharded meshing stage."""
    _LocalTaskQueue, tc = _load_meshing_modules()
    mesh_output_dir = os.path.join(
        cloudvolume_path.replace("file://", ""), mesh_dir
    )
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
    _execute_tasks(mesh_tasks)
    logger.info("Mesh fragments generated.")
    return mesh_output_dir


def finalize_sharded_meshes(cloudvolume_path, mesh_dir):
    """Run the second-pass sharded multi-resolution mesh stage."""
    _LocalTaskQueue, tc = _load_meshing_modules()
    mesh_output_dir = os.path.join(
        cloudvolume_path.replace("file://", ""), mesh_dir
    )
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
    _execute_tasks(multires_tasks)
    logger.info("Multi-resolution meshes generated.")

    for f in glob.glob(os.path.join(mesh_output_dir, "*.frags")):
        os.remove(f)
    return mesh_output_dir


def generate_meshes(cloudvolume_path, mesh_dir, unsharded):
    """Run the full igneous mesh generation pipeline on an existing CloudVolume."""
    if unsharded:
        forge_unsharded_mesh_fragments(cloudvolume_path, mesh_dir)
        return finalize_unsharded_meshes(cloudvolume_path, mesh_dir)

    forge_sharded_mesh_fragments(cloudvolume_path, mesh_dir)
    return finalize_sharded_meshes(cloudvolume_path, mesh_dir)
