import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import build_label_names_for_inputs, parse_labels, remap_labels_sparse


class BuildLabelNamesForInputsTest(unittest.TestCase):
    def test_tiff_labels_include_all_sparse_ids(self):
        data = np.array([[[0, 2], [17, 401]]], dtype=np.uint32)

        label_names = build_label_names_for_inputs(
            ["/tmp/my_segmentation.tif"],
            data,
        )

        self.assertEqual(
            label_names,
            {
                2: "my_segmentation_label_2",
                17: "my_segmentation_label_17",
                401: "my_segmentation_label_401",
            },
        )

    def test_tiff_label_override_only_updates_requested_id(self):
        data = np.array([[[0, 2], [17, 401]]], dtype=np.uint32)

        label_names = build_label_names_for_inputs(
            ["/tmp/my_segmentation.tif"],
            data,
        )
        label_names.update(parse_labels(["17:fan_shaped_body"]))

        self.assertEqual(label_names[2], "my_segmentation_label_2")
        self.assertEqual(label_names[17], "fan_shaped_body")
        self.assertEqual(label_names[401], "my_segmentation_label_401")

    def test_stl_labels_use_filenames(self):
        label_names = build_label_names_for_inputs(
            ["/tmp/PB.stl", "/tmp/EB.stl"],
            np.zeros((1, 1, 1), dtype=np.uint32),
        )

        self.assertEqual(label_names, {1: "PB", 2: "EB"})


class RemapLabelsSparseTest(unittest.TestCase):
    def test_remaps_sparse_large_label_ids(self):
        data = np.array([[[0, 1], [500_000_000, 500_000_000]]], dtype=np.uint32)

        remapped = remap_labels_sparse(
            data,
            {1: 7, 500_000_000: 9},
        )

        expected = np.array([[[0, 7], [9, 9]]], dtype=np.uint32)
        self.assertTrue(np.array_equal(remapped, expected))

    def test_unmapped_labels_become_zero(self):
        data = np.array([[[0, 1], [2, 99]]], dtype=np.uint32)

        remapped = remap_labels_sparse(data, {1: 4, 2: 5})

        expected = np.array([[[0, 4], [5, 0]]], dtype=np.uint32)
        self.assertTrue(np.array_equal(remapped, expected))


if __name__ == "__main__":
    unittest.main()
