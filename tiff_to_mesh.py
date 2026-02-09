import argparse
import logging
import os
import shutil
import subprocess

import numpy as np
import tifffile
from cloudvolume import CloudVolume
from taskqueue import LocalTaskQueue

import igneous.task_creation as tc

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--d', required=True, help='Directory containing TIFF file')
parser.add_argument('--out', default=None, help='Output directory (default: same as input)')
parser.add_argument('--res', nargs=3, type=int, default=[8, 8, 42], metavar=('X', 'Y', 'Z'), help='Resolution (default: 8 8 42)')
parser.add_argument('--unsharded', action='store_true', help='Use unsharded format (default: sharded)')
parser.add_argument('--setgit', action='store_true', help='Initialize git repo in output directory')
args = parser.parse_args()

TIFF_PATH = args.d
OUTPUT_PATH = args.out if args.out else args.d
RESOLUTION = tuple(args.res)
CHUNK_SIZE = (64, 64, 64)
MESH_DIR = "mesh"
UNSHARDED = args.unsharded
UINT32_MAX = np.iinfo(np.uint32).max


def ensure_uint32_labels(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint32:
        return array

    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(np.uint32, copy=False)

    if np.issubdtype(array.dtype, np.floating):
        # Validate in chunks to reduce memory usage for large 3D volumes
        # Process array in slabs to avoid allocating large temporary arrays
        chunk_size = 1024 * 1024 * 10  # ~10M elements per chunk
        flat_view = array.ravel()
        total_elements = flat_view.size
        
        min_val = np.inf
        max_val = -np.inf
        
        for start_idx in range(0, total_elements, chunk_size):
            end_idx = min(start_idx + chunk_size, total_elements)
            chunk = flat_view[start_idx:end_idx]
            
            # Check for NaN/inf in chunk
            if not np.isfinite(chunk).all():
                raise ValueError("Input contains NaN or infinite values; cannot convert to uint32 labels safely.")
            
            # Check integrality in chunk without allocating floor array
            if (chunk != np.trunc(chunk)).any():
                raise ValueError("Input contains non-integer float values; cannot convert to uint32 labels safely.")
            
            # Track min/max
            chunk_min = chunk.min()
            chunk_max = chunk.max()
            min_val = min(min_val, chunk_min)
            max_val = max(max_val, chunk_max)
        
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

if os.path.isfile(TIFF_PATH):
    tiff_file = TIFF_PATH
else:
    tiff_files = [f for f in os.listdir(TIFF_PATH) if f.endswith('.tif') or f.endswith('.tiff')]
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {TIFF_PATH}")
    tiff_file = os.path.join(TIFF_PATH, tiff_files[0])

output_dir = os.path.join(OUTPUT_PATH, "output_volume")
mesh_output_dir = os.path.join(output_dir, MESH_DIR)
cloudvolume_path = f"file://{output_dir}"

if os.path.exists(mesh_output_dir):
    shutil.rmtree(mesh_output_dir)

data = tifffile.imread(tiff_file)
logger.info(f"Loaded shape: {data.shape}, dtype: {data.dtype}")

data = ensure_uint32_labels(data)

if data.ndim == 3:
    data = np.transpose(data, (2, 1, 0))

os.makedirs(output_dir, exist_ok=True)

info = {
    "data_type": "uint32",
    "num_channels": 1,
    "type": "segmentation",
    "scales": [{
        "key": f"{RESOLUTION[0]}_{RESOLUTION[1]}_{RESOLUTION[2]}",
        "resolution": list(RESOLUTION),
        "size": list(data.shape),
        "voxel_offset": [0, 0, 0],
        "chunk_sizes": [list(CHUNK_SIZE)],
        "encoding": "raw"
    }]
}

import json
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
    for pattern in ["*.frags", "*.spatial.gz"]:
        for f in glob.glob(os.path.join(mesh_output_dir, pattern)):
            os.remove(f)

cv = CloudVolume(cloudvolume_path)
logger.info(f"Mesh info: {cv.mesh.meta.info}")

if args.setgit:
    git_dir = os.path.join(OUTPUT_PATH, ".git")
    if not os.path.exists(git_dir):
        subprocess.run(["git", "init"], cwd=OUTPUT_PATH, capture_output=True)
        logger.info(f"Git repo initialized in {OUTPUT_PATH}")

logger.info("Done!")
