import argparse
import logging
import os
import shutil
import subprocess
import json

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
parser.add_argument('--res', nargs=3, type=int, default=[800, 800, 840], metavar=('X', 'Y', 'Z'), help='Output resolution in nm for aligned meshes (default: 800 800 840)')
parser.add_argument('--unsharded', action='store_true', help='Use unsharded format (default: sharded)')
parser.add_argument('--setgit', action='store_true', help='Initialize git repo in output directory')
args = parser.parse_args()

TIFF_PATH = args.d
OUTPUT_PATH = args.out if args.out else (os.path.dirname(args.d) if os.path.isfile(args.d) else args.d)
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

if args.setgit:
    git_dir = os.path.join(OUTPUT_PATH, ".git")
    if not os.path.exists(git_dir):
        subprocess.run(["git", "init"], cwd=OUTPUT_PATH, capture_output=True)
        logger.info(f"Git repo initialized in {OUTPUT_PATH}")

logger.info("Done!")
