import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import build_label_names_for_inputs, parse_label_csv, parse_labels, remap_labels_sparse


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


class ParseLabelCsvTest(unittest.TestCase):
    def _write_csv(self, lines, encoding="utf-8"):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding=encoding)
        f.write("\n".join(lines))
        f.close()
        return f.name

    def test_basic_csv(self):
        path = self._write_csv(["1,ellipsoid_body", "2,fan_shaped_body"])
        result = parse_label_csv(path)
        self.assertEqual(result, {1: "ellipsoid_body", 2: "fan_shaped_body"})

    def test_header_row_skipped(self):
        path = self._write_csv(["id,name", "1,ellipsoid_body", "17,protocerebral_bridge"])
        result = parse_label_csv(path)
        self.assertEqual(result, {1: "ellipsoid_body", 17: "protocerebral_bridge"})

    def test_utf8_bom_is_ignored(self):
        path = self._write_csv(["1,ellipsoid_body", "17,protocerebral_bridge"], encoding="utf-8-sig")
        result = parse_label_csv(path)
        self.assertEqual(result, {1: "ellipsoid_body", 17: "protocerebral_bridge"})

    def test_semicolon_delimiter_is_supported(self):
        path = self._write_csv(["id;name", "1;ellipsoid_body", "17;protocerebral_bridge"])
        result = parse_label_csv(path)
        self.assertEqual(result, {1: "ellipsoid_body", 17: "protocerebral_bridge"})

    def test_reordered_header_columns_are_supported(self):
        path = self._write_csv(
            ["name,notes,segment_id", "ellipsoid_body,core,1", "protocerebral_bridge,midline,17"]
        )
        result = parse_label_csv(path)
        self.assertEqual(result, {1: "ellipsoid_body", 17: "protocerebral_bridge"})

    def test_cp1252_spreadsheet_export_is_supported(self):
        path = self._write_csv(["id,name", "1,K\xe4fer"], encoding="cp1252")
        result = parse_label_csv(path)
        self.assertEqual(result, {1: "K\xe4fer"})

    def test_comments_and_blank_lines_skipped(self):
        path = self._write_csv(["# this is a comment", "", "1,PB", "", "# another", "2,EB"])
        result = parse_label_csv(path)
        self.assertEqual(result, {1: "PB", 2: "EB"})

    def test_whitespace_stripped(self):
        path = self._write_csv(["  1 , ellipsoid_body  ", " 2 ,fan_shaped_body"])
        result = parse_label_csv(path)
        self.assertEqual(result, {1: "ellipsoid_body", 2: "fan_shaped_body"})

    def test_sparse_ids(self):
        path = self._write_csv(["3,alpha", "401,beta"])
        result = parse_label_csv(path)
        self.assertEqual(result, {3: "alpha", 401: "beta"})

    def test_missing_name_column_raises(self):
        path = self._write_csv(["1"])
        with self.assertRaises(ValueError):
            parse_label_csv(path)

    def test_empty_file(self):
        path = self._write_csv([""])
        result = parse_label_csv(path)
        self.assertEqual(result, {})

    def test_label_file_overrides_auto_names(self):
        data = np.array([[[0, 2], [17, 401]]], dtype=np.uint32)
        label_names = build_label_names_for_inputs(["/tmp/seg.tif"], data)
        path = self._write_csv(["17,protocerebral_bridge"])
        label_names.update(parse_label_csv(path))
        self.assertEqual(label_names[2], "seg_label_2")
        self.assertEqual(label_names[17], "protocerebral_bridge")
        self.assertEqual(label_names[401], "seg_label_401")


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
