import argparse
import logging
import os
import shutil
import subprocess
import json
from dataclasses import dataclass

import numpy as np
import tifffile
import trimesh
from cloudvolume import CloudVolume
from taskqueue import LocalTaskQueue

import igneous.task_creation as tc

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


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


def push_to_github(output_dir: str, repo_name: str):
    username = get_github_username()
    full_repo = f"{username}/{repo_name}"
    logger.info(f"Pushing to GitHub repo: {full_repo}")

    # Initialize git repo in output_dir with explicit branch name
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=output_dir,
        capture_output=True, text=True, check=True
    )

    # Create the repo (tolerate "already exists")
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

    # Stage and commit
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

    # Set remote origin (replace if already set)
    remote_url = f"https://github.com/{full_repo}.git"
    subprocess.run(
        ["git", "remote", "remove", "origin"], cwd=output_dir,
        capture_output=True  # OK to fail if no origin yet
    )
    subprocess.run(
        ["git", "remote", "add", "origin", remote_url], cwd=output_dir,
        capture_output=True, text=True, check=True
    )

    # Push to remote
    push_result = subprocess.run(
        ["git", "push", "-u", "origin", "main"],
        cwd=output_dir, capture_output=True, text=True
    )
    if push_result.returncode != 0:
        raise RuntimeError(
            f"Failed to push to GitHub:\n{push_result.stderr.strip()}"
        )

    # Get commit hash
    hash_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=output_dir,
        capture_output=True, text=True, check=True
    )
    commit_hash = hash_result.stdout.strip()

    raw_link = (
        f"https://raw.githubusercontent.com/{full_repo}/"
        f"{commit_hash}/mesh/|neuroglancer-precomputed:"
    )
    logger.info(f"Neuroglancer source URL:\n{raw_link}")


parser = argparse.ArgumentParser()
parser.add_argument('--d', required=True, help='Directory or path to a TIFF/STL file')
parser.add_argument(
    '--out',
    default=None,
    help='Base output directory; files are written to <out>/<input_name> (default: same as input directory)'
)
parser.add_argument('--res', nargs=3, type=int, default=[800, 800, 840], metavar=('X', 'Y', 'Z'), help='Output resolution in nm for aligned meshes (default: 800 800 840)')
parser.add_argument('--unsharded', action='store_true', help='Use unsharded format (default: sharded)')
git_group = parser.add_mutually_exclusive_group()
git_group.add_argument('--setgit', action='store_true', help='Initialize git repo in output directory')
git_group.add_argument('--push', metavar='REPO_NAME', default=None,
                       help='Create a public GitHub repo, push output, and print Neuroglancer raw link (implies --setgit)')
args = parser.parse_args()

INPUT_PATH = args.d
OUTPUT_PATH = args.out if args.out else (os.path.dirname(args.d) if os.path.isfile(args.d) else args.d)
RESOLUTION = tuple(args.res)
MESH_DIR = "mesh"
UNSHARDED = args.unsharded
UINT32_MAX = np.iinfo(np.uint32).max



def ensure_uint32_labels(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint32:
        return array

    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(np.uint32, copy=False)

    # Handle empty arrays early to avoid crashes from min()/max() calls
    if array.size == 0:
        return array.astype(np.uint32, copy=False)

    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("Input contains NaN or infinite values; cannot convert to uint32 labels safely.")
        if not np.equal(array, np.floor(array)).all():
            raise ValueError("Input contains non-integer float values; cannot convert to uint32 labels safely.")
        min_val = array.min()
        max_val = array.max()
        if min_val < 0 or max_val > UINT32_MAX:
            raise ValueError(
                f"Input values out of uint32 range [{0}, {UINT32_MAX}]: min={min_val}, max={max_val}"
            )
        logger.warning("Converting integer-valued float labels to uint32.")
        return array.astype(np.uint32, copy=False)

    if np.issubdtype(array.dtype, np.integer):
        min_val = array.min()
        max_val = array.max()
        if min_val < 0 or max_val > UINT32_MAX:
            raise ValueError(
                f"Input values out of uint32 range [{0}, {UINT32_MAX}]: min={min_val}, max={max_val}"
            )
        return array.astype(np.uint32, copy=False)

    raise TypeError(f"Unsupported dtype for segmentation labels: {array.dtype}")

@dataclass
class MeshEntryLabels:
    """Load TIFF or STL file(s) into a uint32 voxel array (X,Y,Z) and
    compute Neuroglancer volume parameters (chunk size, voxel offset).

    The chunk size is chosen per-axis as the largest power-of-2 candidate
    that still yields at least `min_chunks` chunks along that axis.

    Voxel offset:
      TIFF — centers the volume at the origin (-shape // 2).
      STL  — preserves the original physical position (global_min in voxel coords).
    """
    file_paths: list
    resolution: tuple
    min_chunks: int = 8

    _CHUNK_CANDIDATES = (32, 64, 128)

    def __post_init__(self):
        exts = {os.path.splitext(f)[1].lower() for f in self.file_paths}
        if exts <= {'.tif', '.tiff'}:
            if len(self.file_paths) != 1:
                raise ValueError("Only one TIFF file supported at a time")
            self.data = self._load_tiff(self.file_paths[0])
            self.voxel_offset = [-(s // 2) for s in self.data.shape]
        elif exts <= {'.stl'}:
            self.data = self._load_stls()
            # voxel_offset set in _load_stls from global_min
        else:
            raise ValueError(f"Mixed or unsupported file types: {exts}")

    def _load_tiff(self, path) -> np.ndarray:
        data = tifffile.imread(path)
        logger.info(f"Loaded TIFF shape: {data.shape}, dtype: {data.dtype}")
        data = ensure_uint32_labels(data)
        if data.ndim == 3:
            data = np.transpose(data, (2, 1, 0))
        return data

    def _load_stls(self) -> np.ndarray:
        origins = []
        grids = []
        for stl_file in self.file_paths:
            mesh = trimesh.load(stl_file, force='mesh')
            logger.info(f"Loaded STL {os.path.basename(stl_file)}: "
                        f"{len(mesh.vertices)} verts, {len(mesh.faces)} faces")
            mesh.vertices /= np.array(self.resolution, dtype=float)
            vg = mesh.voxelized(pitch=1.0).fill()
            origin = np.round(vg.transform[:3, 3]).astype(int)
            origins.append(origin)
            grids.append(vg.matrix)

        # Compute global bounding box across all voxelized meshes
        global_min = np.min(origins, axis=0)
        global_max = np.max(
            [o + np.array(g.shape) for o, g in zip(origins, grids)], axis=0
        )
        volume_shape = tuple(global_max - global_min)
        self.voxel_offset = global_min.tolist()

        # Paint each mesh into the combined volume with a unique label
        data = np.zeros(volume_shape, dtype=np.uint32)
        for label_id, (origin, grid) in enumerate(zip(origins, grids), start=1):
            offset = origin - global_min
            slices = tuple(slice(o, o + s) for o, s in zip(offset, grid.shape))
            data[slices][grid] = np.uint32(label_id)

        logger.info(f"Combined {len(self.file_paths)} STL files into "
                    f"volume {volume_shape}, labels 1-{len(self.file_paths)}")
        return data

    def compute_chunk_size(self) -> tuple:
        """Pick the largest power-of-2 chunk per axis such that
        each axis has at least `min_chunks` chunks."""
        chunks = []
        for dim_size in self.data.shape:
            chunk = self._CHUNK_CANDIDATES[0]
            for candidate in self._CHUNK_CANDIDATES:
                if dim_size >= candidate * self.min_chunks:
                    chunk = candidate
            chunks.append(chunk)
        return tuple(chunks)

    def compute_translation_nm(self) -> tuple:
        """Physical translation in nm (for reference / verification).

        With voxel_offset baked into info.json this should no longer be
        needed for manual Neuroglancer source transforms, but kept for
        redundancy/checks.
        """
        return tuple(o * r for o, r in zip(self.voxel_offset, self.resolution))

    def build_info(self) -> dict:
        """Build the Neuroglancer precomputed info dict using
        the computed chunk size and voxel offset."""
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
            "scales": [{
                "key": f"{self.resolution[0]}_{self.resolution[1]}_{self.resolution[2]}",
                "resolution": list(self.resolution),
                "size": list(self.data.shape),
                "voxel_offset": list(self.voxel_offset),
                "chunk_sizes": [list(chunk_size)],
                "encoding": "raw",
            }]
        }

SUPPORTED_EXTENSIONS = ('.tif', '.tiff', '.stl')

if os.path.isfile(INPUT_PATH):
    input_files = [INPUT_PATH]
else:
    candidates = sorted(f for f in os.listdir(INPUT_PATH)
                        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS)
    if not candidates:
        raise ValueError(f"No TIFF or STL files found in {INPUT_PATH}")
    input_files = [os.path.join(INPUT_PATH, f) for f in candidates]

if os.path.isfile(INPUT_PATH):
    volume_name = os.path.splitext(os.path.basename(INPUT_PATH))[0]
else:
    volume_name = os.path.basename(os.path.normpath(INPUT_PATH))

if args.out:
    output_dir = os.path.join(args.out, volume_name)
elif os.path.isfile(INPUT_PATH):
    output_dir = os.path.join(os.path.dirname(INPUT_PATH), volume_name)
else:
    output_dir = INPUT_PATH
mesh_output_dir = os.path.join(output_dir, MESH_DIR)
cloudvolume_path = f"file://{output_dir}"

if os.path.exists(mesh_output_dir):
    shutil.rmtree(mesh_output_dir)

entry = MeshEntryLabels(file_paths=input_files, resolution=RESOLUTION)
data = entry.data
info = entry.build_info()

os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "info"), "w") as f:
    json.dump(info, f)

cv = CloudVolume(cloudvolume_path, compress=False)
cv[:] = data[:]

logger.info(f"Volume shape: {cv.shape}, bounds: {cv.bounds}")

tq = LocalTaskQueue(parallel=4)

if UNSHARDED:
    logger.info("Step 1: Creating unsharded mesh fragments...")
    mesh_tasks = tc.create_meshing_tasks(
        layer_path=cloudvolume_path,
        mip=0,
        shape=(256, 256, 256),
        simplification=True,
        max_simplification_error=40,
        mesh_dir=MESH_DIR,
        sharded=False,
        spatial_index=False,
    )
    tq.insert(mesh_tasks)
    tq.execute()
    logger.info("Mesh fragments generated.")

    logger.info("Step 2: Creating unsharded multi-resolution draco meshes...")
    multires_tasks = tc.create_unsharded_multires_mesh_tasks(
        cloudpath=cloudvolume_path,
        num_lod=2,
        mesh_dir=MESH_DIR,
        vertex_quantization_bits=16,
        min_chunk_size=(128, 128, 128),
    )
    tq.insert(multires_tasks)
    tq.execute()
    logger.info("Multi-resolution meshes generated.")

    import glob
    for pattern in ["*:*"]:
        for f in glob.glob(os.path.join(mesh_output_dir, pattern)):
            os.remove(f)
else:
    logger.info("Step 1: Creating sharded mesh fragments...")
    mesh_tasks = tc.create_meshing_tasks(
        layer_path=cloudvolume_path,
        mip=0,
        shape=(256, 256, 256),
        simplification=True,
        max_simplification_error=40,
        mesh_dir=MESH_DIR,
        sharded=True,
        spatial_index=True,
        compress="gzip",
    )
    tq.insert(mesh_tasks)
    tq.execute()
    logger.info("Mesh fragments generated.")

    logger.info("Step 2: Creating sharded multi-resolution draco meshes...")
    multires_tasks = tc.create_sharded_multires_mesh_tasks(
        cloudpath=cloudvolume_path,
        num_lod=2,
        mesh_dir=MESH_DIR,
        vertex_quantization_bits=16,
        min_chunk_size=(128, 128, 128),
        draco_compression_level=7,
        shard_index_bytes=2**13,
        minishard_index_bytes=2**15,
    )
    tq.insert(multires_tasks)
    tq.execute()
    logger.info("Multi-resolution meshes generated.")

    import glob
    for pattern in ["*.frags"]:
        for f in glob.glob(os.path.join(mesh_output_dir, pattern)):
            os.remove(f)

cv = CloudVolume(cloudvolume_path)
logger.info(f"Mesh info: {cv.mesh.meta.info}")

if args.push:
    push_to_github(output_dir, args.push)
elif args.setgit:
    git_dir = os.path.join(output_dir, ".git")
    if not os.path.exists(git_dir):
        subprocess.run(["git", "init"], cwd=output_dir, capture_output=True)
        logger.info(f"Git repo initialized in {output_dir}")

logger.info("Done!")
