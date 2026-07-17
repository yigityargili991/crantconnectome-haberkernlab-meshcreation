import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from contextlib import ExitStack, redirect_stderr

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class PublicNamespaceTest(unittest.TestCase):
    def test_expected_library_api_is_exported(self):
        import mesh_creation

        expected = {
            "MeshEntryLabels",
            "MeshConverter",
            "DatastackMerger",
            "tiff_to_mesh",
            "merge_datastacks",
            "replace_labels",
        }

        self.assertTrue(
            expected.issubset(set(dir(mesh_creation))),
            f"missing public exports: {sorted(expected - set(dir(mesh_creation)))}",
        )
        for name in expected:
            self.assertTrue(callable(getattr(mesh_creation, name)), name)

    def test_replace_labels_has_the_documented_leading_parameters(self):
        from mesh_creation import replace_labels

        parameters = list(inspect.signature(replace_labels).parameters.values())
        self.assertGreaterEqual(len(parameters), 5)
        self.assertEqual(
            [parameter.name for parameter in parameters[:5]],
            [
                "base",
                "replacement",
                "labels",
                "output_dir",
                "replacement_labels",
            ],
        )
        self.assertIsNone(parameters[4].default)

    def test_public_parser_aliases_parse_the_legacy_cli_shapes(self):
        from mesh_creation import build_merge_parser, build_tiff_parser

        tiff_args = build_tiff_parser().parse_args(
            [
                "--d",
                "input.tif",
                "--out",
                "meshes",
                "--res",
                "8",
                "8",
                "42",
                "--voxel-offset",
                "0",
                "0",
                "0",
                "--unsharded",
                "--labels",
                "1:PB",
            ]
        )
        self.assertEqual(tiff_args.d, "input.tif")
        self.assertEqual(tiff_args.out, "meshes")
        self.assertEqual(tiff_args.res, [8, 8, 42])
        self.assertEqual(tiff_args.voxel_offset, [0, 0, 0])
        self.assertTrue(tiff_args.unsharded)
        self.assertEqual(tiff_args.labels, ["1:PB"])

        merge_args = build_merge_parser().parse_args(
            [
                "stack-a",
                "stack-b",
                "--out",
                "merged",
                "--unsharded",
                "--exclude",
                "stack-a",
                "1",
            ]
        )
        self.assertEqual(merge_args.datastacks, ["stack-a", "stack-b"])
        self.assertEqual(merge_args.out, "merged")
        self.assertTrue(merge_args.unsharded)
        self.assertEqual(merge_args.exclude, ["stack-a", "1"])

    def test_tiff_parser_keeps_setgit_and_push_mutually_exclusive(self):
        from mesh_creation import build_tiff_parser

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_tiff_parser().parse_args(
                ["--d", "input.tif", "--setgit", "--push", "repo"]
            )


class LegacyModuleImportTest(unittest.TestCase):
    def test_importing_tiff_to_mesh_does_not_parse_arguments_or_run_conversion(self):
        # Do this in a fresh interpreter because another test importing the package
        # must not be able to hide import-time behavior through sys.modules caching.
        probe = """
import argparse

def forbidden_parse(*args, **kwargs):
    raise AssertionError("tiff_to_mesh parsed CLI arguments during import")

argparse.ArgumentParser.parse_args = forbidden_parse
import tiff_to_mesh
print("import-ok")
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            files_created = list(Path(cwd).iterdir())

        self.assertEqual(
            result.returncode,
            0,
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertEqual(result.stdout.strip(), "import-ok")
        self.assertEqual(files_created, [])


class CliCompatibilityTest(unittest.TestCase):
    def _help(self, script_name):
        return subprocess.run(
            [sys.executable, str(ROOT / script_name), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_tiff_cli_retains_existing_options(self):
        result = self._help("tiff_to_mesh.py")

        self.assertEqual(result.returncode, 0, result.stderr)
        for option in (
            "--d",
            "--out",
            "--res",
            "--voxel-offset",
            "--unsharded",
            "--labels",
            "--label-file",
            "--setgit",
            "--push",
        ):
            with self.subTest(option=option):
                self.assertIn(option, result.stdout)

    def test_merge_cli_retains_existing_arguments_and_options(self):
        result = self._help("merge_datastacks.py")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("datastacks", result.stdout.lower())
        for option in (
            "--out",
            "--unsharded",
            "--labels",
            "--exclude",
            "--setgit",
            "--push",
        ):
            with self.subTest(option=option):
                self.assertIn(option, result.stdout)

    def test_tiff_main_delegates_the_legacy_arguments_to_the_library_workflow(self):
        from mesh_creation import conversion

        delegate = mock.Mock()
        with mock.patch.object(conversion, "tiff_to_mesh", delegate):
            conversion.main(
                [
                    "--d",
                    "input.tif",
                    "--out",
                    "meshes",
                    "--res",
                    "8",
                    "8",
                    "42",
                    "--voxel-offset",
                    "0",
                    "0",
                    "0",
                    "--unsharded",
                    "--labels",
                    "1:PB",
                ]
            )

        delegate.assert_called_once_with(
            input_path="input.tif",
            output_dir="meshes",
            resolution=[8, 8, 42],
            voxel_offset=[0, 0, 0],
            unsharded=True,
            labels=["1:PB"],
            label_file=None,
            setgit=False,
            push=None,
        )

    def test_merge_main_delegates_the_legacy_arguments_to_the_library_workflow(self):
        from mesh_creation import merging

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            first.mkdir()
            second.mkdir()
            delegate = mock.Mock(
                return_value={
                    str(first.resolve()): {1: 1},
                    str(second.resolve()): {2: 2},
                }
            )
            with mock.patch.object(merging, "merge_datastacks", delegate):
                merging.main(
                    [
                        str(first),
                        str(second),
                        "--out",
                        str(Path(tmp) / "merged"),
                        "--unsharded",
                        "--exclude",
                        first.name,
                        "1",
                    ]
                )

        delegate.assert_called_once_with(
            datastack_dirs=[str(first), str(second)],
            output_dir=str(Path(tmp) / "merged"),
            mesh_dir="mesh",
            unsharded=True,
            labels=None,
            exclude={os.path.abspath(first): {1}},
            source_properties={
                os.path.abspath(first): None,
                os.path.abspath(second): None,
            },
            validate_labels=False,
        )

    def test_merge_cli_rejects_grouped_options_with_ambiguous_basenames(self):
        from mesh_creation import merging

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a" / "stack"
            second = Path(tmp) / "b" / "stack"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            delegate = mock.Mock()
            with mock.patch.object(merging, "merge_datastacks", delegate):
                with redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit):
                        merging.main(
                            [
                                str(first),
                                str(second),
                                "--out",
                                str(Path(tmp) / "merged"),
                                "--exclude",
                                "stack",
                                "1",
                            ]
                        )

        delegate.assert_not_called()
        self.assertIn("ambiguous", stderr.getvalue())


class LegacyAdapterTest(unittest.TestCase):
    def test_omitted_source_properties_defer_to_the_library_default(self):
        import merge_datastacks as legacy

        delegate = mock.Mock(return_value={})
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            first.mkdir()
            second.mkdir()
            with mock.patch.object(legacy, "_merge_datastacks", delegate):
                legacy.merge_datastacks(
                    [str(first), str(second)], str(Path(tmp) / "merged")
                )

        self.assertIsNone(delegate.call_args.kwargs["source_properties"])

    def test_supplied_source_properties_expand_to_absolute_paths(self):
        import merge_datastacks as legacy

        delegate = mock.Mock(return_value={})
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            first.mkdir()
            second.mkdir()
            with mock.patch.object(legacy, "_merge_datastacks", delegate):
                legacy.merge_datastacks(
                    [str(first), str(second)],
                    str(Path(tmp) / "merged"),
                    source_properties={"first": {1: "PB"}},
                )

        self.assertEqual(
            delegate.call_args.kwargs["source_properties"],
            {os.path.abspath(str(first)): {1: "PB"}},
        )


class ReplaceLabelsTest(unittest.TestCase):
    def _write_properties(self, stack, labels):
        properties_dir = stack / "segment_properties"
        properties_dir.mkdir(parents=True)
        info = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids": [str(label_id) for label_id in labels],
                "properties": [
                    {
                        "id": "label",
                        "type": "label",
                        "values": list(labels.values()),
                    }
                ],
            },
        }
        (properties_dir / "info").write_text(json.dumps(info), encoding="utf-8")

    def _call_with_fake_merge(
        self,
        *,
        base_labels,
        replacement_labels,
        selected_base,
        selected_replacement=None,
        **options,
    ):
        from mesh_creation import merge_datastacks, replace_labels

        calls = []
        result_sentinel = object()

        def fake_merge(*args, **kwargs):
            calls.append((args, kwargs))
            return result_sentinel

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base_stack"
            replacement = tmp / "replacement_stack"
            output = tmp / "merged"
            base.mkdir()
            replacement.mkdir()
            self._write_properties(base, base_labels)
            self._write_properties(replacement, replacement_labels)

            labels_by_path = {
                str(base.resolve()): set(base_labels),
                str(replacement.resolve()): set(replacement_labels),
            }

            def fake_present_labels(source_path):
                return labels_by_path[str(Path(source_path).resolve())]

            # A replacement operation is deliberately specified as a merge plan.
            # Patching the global used by replace_labels proves there is one
            # implementation of the actual merge behavior.
            self.assertIn("merge_datastacks", replace_labels.__globals__)
            self.assertIn("_read_present_labels", replace_labels.__globals__)
            with mock.patch.dict(
                replace_labels.__globals__,
                {
                    "merge_datastacks": fake_merge,
                    "_read_present_labels": fake_present_labels,
                },
            ):
                result = replace_labels(
                    base,
                    replacement,
                    selected_base,
                    output,
                    replacement_labels=selected_replacement,
                    **options,
                )

            self.assertIs(result, result_sentinel)
            self.assertEqual(len(calls), 1)
            args, kwargs = calls[0]
            bound = inspect.signature(merge_datastacks).bind_partial(*args, **kwargs)
            arguments = bound.arguments

            delegated_paths = list(arguments["datastack_dirs"])
            source_paths = [Path(path).resolve() for path in delegated_paths]
            self.assertEqual(source_paths, [base.resolve(), replacement.resolve()])
            self.assertEqual(Path(arguments["output_dir"]).resolve(), output.resolve())

            return arguments, delegated_paths[0], delegated_paths[1]

    def test_replaces_selected_base_labels_with_all_of_replacement(self):
        arguments, base_path, replacement_path = self._call_with_fake_merge(
            base_labels={1: "alpha", 2: "beta", 3: "gamma"},
            replacement_labels={10: "new-alpha", 20: "new-beta"},
            selected_base=["alpha", 2],
            unsharded=True,
        )

        exclusions = arguments["exclude"]
        self.assertEqual(exclusions[base_path], {1, 2})
        self.assertIsNone(arguments.get("include"))
        self.assertTrue(arguments["unsharded"])

    def test_replacement_selection_includes_only_selected_replacement_labels(self):
        arguments, base_path, replacement_path = self._call_with_fake_merge(
            base_labels={1: "alpha", 2: "beta"},
            replacement_labels={
                10: "candidate-a",
                20: "candidate-b",
                30: "candidate-c",
            },
            selected_base=[1],
            selected_replacement=["candidate-b", 30],
        )

        exclusions = arguments["exclude"]
        self.assertEqual(exclusions[base_path], {1})
        self.assertEqual(arguments["include"], {replacement_path: {20, 30}})

    def test_rejects_labels_that_are_not_present_before_delegating(self):
        from mesh_creation import replace_labels

        delegate = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base"
            replacement = tmp / "replacement"
            output = tmp / "output"
            base.mkdir()
            replacement.mkdir()
            self._write_properties(base, {1: "alpha"})
            self._write_properties(replacement, {2: "new-alpha"})

            with mock.patch.dict(
                replace_labels.__globals__,
                {
                    "merge_datastacks": delegate,
                    "_read_present_labels": lambda path: (
                        {1} if Path(path).name == "base" else {2}
                    ),
                },
            ):
                with self.assertRaisesRegex(ValueError, "not present"):
                    replace_labels(base, replacement, [99], output)

        delegate.assert_not_called()

    def test_rejects_exclusions_that_remove_every_replacement_label(self):
        from mesh_creation import replace_labels

        delegate = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base"
            replacement = tmp / "replacement"
            base.mkdir()
            replacement.mkdir()
            self._write_properties(base, {1: "alpha"})
            self._write_properties(replacement, {2: "new-alpha"})

            with mock.patch.dict(
                replace_labels.__globals__,
                {
                    "merge_datastacks": delegate,
                    "_read_present_labels": lambda path: (
                        {1} if Path(path).name == "base" else {2}
                    ),
                },
            ):
                with self.assertRaisesRegex(ValueError, "No replacement labels"):
                    replace_labels(
                        base,
                        replacement,
                        [1],
                        tmp / "output",
                        exclude={replacement: [2]},
                    )

        delegate.assert_not_called()


class ConversionBehaviorTest(unittest.TestCase):
    def _run_converter(self, *, unsharded):
        from mesh_creation import conversion

        data = np.array([[[0, 2]]], dtype=np.uint32)
        base_info = {
            "data_type": "uint32",
            "num_channels": 1,
            "type": "segmentation",
            "scales": [
                {
                    "key": "8_8_42",
                    "resolution": [8, 8, 42],
                    "size": [1, 1, 2],
                    "voxel_offset": [0, 0, 0],
                    "chunk_sizes": [[32, 32, 32]],
                    "encoding": "raw",
                }
            ],
        }
        entry = SimpleNamespace(data=data, build_info=mock.Mock(return_value=base_info))
        entry_type = mock.Mock(return_value=entry)
        write_volume = mock.MagicMock()
        write_volume.shape = (1, 1, 2, 1)
        write_volume.bounds = "test-bounds"
        inspect_volume = SimpleNamespace(
            mesh=SimpleNamespace(meta=SimpleNamespace(info={"mesh": "metadata"}))
        )
        cloud_volume = mock.Mock(side_effect=[write_volume, inspect_volume])
        queue = mock.Mock()
        queue_type = mock.Mock(return_value=queue)
        mesh_tasks = object()
        multires_tasks = object()

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "sample.tif"
            output_base = Path(tmp) / "meshes"
            input_path.write_bytes(b"fixture")
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(conversion, "MeshEntryLabels", entry_type)
                )
                stack.enter_context(
                    mock.patch.object(conversion, "CloudVolume", cloud_volume)
                )
                stack.enter_context(
                    mock.patch.object(conversion, "LocalTaskQueue", queue_type)
                )
                create_mesh_tasks = stack.enter_context(
                    mock.patch.object(
                        conversion.tc,
                        "create_meshing_tasks",
                        return_value=mesh_tasks,
                    )
                )
                create_multires_tasks = stack.enter_context(
                    mock.patch.object(
                        conversion.tc,
                        (
                            "create_unsharded_multires_mesh_tasks"
                            if unsharded
                            else "create_sharded_multires_mesh_tasks"
                        ),
                        return_value=multires_tasks,
                    )
                )
                result = conversion.tiff_to_mesh(
                    input_path,
                    output_base,
                    resolution=(8, 8, 42),
                    voxel_offset=(0, 0, 0),
                    unsharded=unsharded,
                    labels={2: "PB"},
                )

            output_dir = output_base / "sample"
            self.assertEqual(result, str(output_dir))
            expected_info = dict(base_info, segment_properties="segment_properties")
            self.assertEqual(
                (output_dir / "info").read_bytes(),
                json.dumps(expected_info).encode(),
            )
            with (output_dir / "segment_properties" / "info").open() as file:
                properties = json.load(file)
            self.assertEqual(properties["inline"]["ids"], ["2"])
            self.assertEqual(properties["inline"]["properties"][0]["values"], ["PB"])

        entry_type.assert_called_once_with(
            file_paths=[str(input_path)],
            resolution=(8, 8, 42),
            voxel_offset_override=(0, 0, 0),
        )
        queue_type.assert_called_once_with(parallel=4)
        create_mesh_tasks.assert_called_once_with(
            layer_path=f"file://{output_dir}",
            mip=0,
            shape=(256, 256, 256),
            simplification=True,
            max_simplification_error=40,
            mesh_dir="mesh",
            sharded=not unsharded,
            spatial_index=not unsharded,
            **({"compress": "gzip"} if not unsharded else {}),
        )
        expected_multires = {
            "cloudpath": f"file://{output_dir}",
            "num_lod": 2,
            "mesh_dir": "mesh",
            "vertex_quantization_bits": 16,
            "min_chunk_size": (128, 128, 128),
        }
        if not unsharded:
            expected_multires.update(
                draco_compression_level=7,
                shard_index_bytes=2**13,
                minishard_index_bytes=2**15,
            )
        create_multires_tasks.assert_called_once_with(**expected_multires)
        self.assertEqual(
            queue.mock_calls,
            [
                mock.call.insert(mesh_tasks),
                mock.call.execute(),
                mock.call.insert(multires_tasks),
                mock.call.execute(),
            ],
        )

    def test_sharded_conversion_keeps_metadata_and_task_contract(self):
        self._run_converter(unsharded=False)

    def test_unsharded_conversion_keeps_metadata_and_task_contract(self):
        self._run_converter(unsharded=True)

    def test_conversion_rejects_mesh_paths_that_can_escape_before_deleting(self):
        from mesh_creation import tiff_to_mesh

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            input_path = tmp / "sample.tif"
            input_path.write_bytes(b"fixture")
            victim = tmp / "victim"
            victim.mkdir()
            marker = victim / "keep-me"
            marker.write_text("safe")

            for mesh_dir in (
                "",
                ".",
                "..",
                "../mesh",
                "nested/../mesh",
                str(victim),
            ):
                with self.subTest(mesh_dir=mesh_dir):
                    with self.assertRaisesRegex(ValueError, "mesh_dir"):
                        tiff_to_mesh(
                            input_path,
                            tmp / "output",
                            mesh_dir=mesh_dir,
                        )
                    self.assertEqual(marker.read_text(), "safe")

    def test_conversion_rejects_a_mesh_symlink_outside_the_output(self):
        from mesh_creation import tiff_to_mesh

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            input_path = tmp / "sample.tif"
            input_path.write_bytes(b"fixture")
            output_dir = tmp / "output" / "sample"
            output_dir.mkdir(parents=True)
            victim = tmp / "victim"
            victim.mkdir()
            marker = victim / "keep-me"
            marker.write_text("safe")
            (output_dir / "mesh").symlink_to(victim, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "mesh_dir"):
                tiff_to_mesh(input_path, tmp / "output")
            self.assertEqual(marker.read_text(), "safe")


class MergePathSafetyTest(unittest.TestCase):
    def test_output_may_not_be_inside_an_input_datastack(self):
        from mesh_creation import merge_datastacks

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            first = tmp / "first"
            second = tmp / "second"
            first.mkdir()
            second.mkdir()

            with self.assertRaisesRegex(ValueError, "Output directory"):
                merge_datastacks([first, second], first / "merged")

    def test_duplicate_input_datastacks_are_rejected_before_meshing(self):
        from mesh_creation import merge_datastacks

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()

            with self.assertRaisesRegex(ValueError, "Duplicate"):
                merge_datastacks([source, source], Path(tmp) / "merged")

    def test_output_may_not_contain_input_datastacks(self):
        from mesh_creation import merge_datastacks

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            first = output / "first"
            second = Path(tmp) / "second"
            first.mkdir(parents=True)
            second.mkdir()

            with self.assertRaisesRegex(ValueError, "Output directory"):
                merge_datastacks([first, second], output)

    def test_duplicate_input_through_symlink_is_rejected(self):
        from mesh_creation import merge_datastacks

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            alias = Path(tmp) / "alias"
            source.mkdir()
            alias.symlink_to(source, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "Duplicate"):
                merge_datastacks([source, alias], Path(tmp) / "merged")

    def test_ambiguous_basename_configuration_is_rejected(self):
        from mesh_creation import merge_datastacks

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a" / "stack"
            second = Path(tmp) / "b" / "stack"
            first.mkdir(parents=True)
            second.mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                merge_datastacks(
                    [first, second],
                    Path(tmp) / "merged",
                    labels={"stack": {1: "PB"}},
                )

    def test_merge_rejects_unsafe_mesh_directory_before_reading_sources(self):
        from mesh_creation import merge_datastacks

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            first.mkdir()
            second.mkdir()

            with self.assertRaisesRegex(ValueError, "mesh_dir"):
                merge_datastacks(
                    [first, second],
                    Path(tmp) / "merged",
                    mesh_dir="../outside",
                )


class MergeLabelPlanTest(unittest.TestCase):
    def test_replacement_plan_keeps_source_order_and_contiguous_ids(self):
        from mesh_creation.merging import _build_label_map

        label_map, skipped = _build_label_map(
            ["stack-a", "stack-b"],
            [[1, 2, 10], [7, 20]],
            {"stack-a": {2}},
            {"stack-b": {7}},
        )

        self.assertEqual(label_map["stack-a"], {1: 1, 10: 2})
        self.assertEqual(label_map["stack-b"], {7: 3})
        self.assertEqual(skipped["stack-a"], [2])
        self.assertEqual(skipped["stack-b"], [20])

    def test_library_validation_rejects_missing_label_ids(self):
        from mesh_creation.merging import _validate_label_selections

        with self.assertRaisesRegex(ValueError, "not present"):
            _validate_label_selections(
                ["stack-a", "stack-b"],
                [[1, 2], [7]],
                {"stack-a": {99}},
                {"stack-b": {7}},
                {},
            )


class ObjectOrientedDelegationTest(unittest.TestCase):
    def test_tiff_function_constructs_the_stateful_converter(self):
        from mesh_creation import MeshConverter, tiff_to_mesh

        sentinel = object()
        run = mock.Mock(return_value=sentinel)
        converter = mock.Mock(run=run)
        constructor = mock.Mock(return_value=converter)

        self.assertIn("MeshConverter", tiff_to_mesh.__globals__)
        with mock.patch.dict(tiff_to_mesh.__globals__, {"MeshConverter": constructor}):
            result = tiff_to_mesh(
                "input.tif",
                output_dir="out",
                resolution=(8, 8, 42),
                voxel_offset=(0, 0, 0),
                unsharded=True,
            )

        self.assertIs(result, sentinel)
        constructor.assert_called_once_with(
            input_path="input.tif",
            output_dir="out",
            resolution=(8, 8, 42),
            voxel_offset=(0, 0, 0),
            unsharded=True,
            labels=None,
            label_file=None,
            setgit=False,
            push=None,
            mesh_dir="mesh",
        )
        run.assert_called_once_with()
        self.assertIs(MeshConverter.convert, MeshConverter.run)

    def test_datastack_merger_delegates_merge_and_replace(self):
        from mesh_creation import DatastackMerger

        merge_result = object()
        replace_result = object()
        merge_delegate = mock.Mock(return_value=merge_result)
        replace_delegate = mock.Mock(return_value=replace_result)
        merger = DatastackMerger(
            ["stack-a", "stack-b"],
            "merged",
            unsharded=True,
            mesh_dir="custom-mesh",
            label_names={"stack-a": {1: "PB"}},
            exclude={"stack-a": [2]},
            source_properties={"stack-a": {1: "PB", 2: "FB"}},
        )
        globals_dict = merger.run.__func__.__globals__

        self.assertIn("merge_datastacks", globals_dict)
        self.assertIn("replace_labels", merger.replace.__func__.__globals__)
        with mock.patch.dict(
            globals_dict,
            {
                "merge_datastacks": merge_delegate,
                "replace_labels": replace_delegate,
            },
        ):
            actual_merge = merger.run()
            actual_replace = merger.replace(
                [1, "PB"],
                replacement_labels=[3],
            )

        self.assertIs(actual_merge, merge_result)
        merge_delegate.assert_called_once_with(
            ["stack-a", "stack-b"],
            "merged",
            mesh_dir="custom-mesh",
            unsharded=True,
            labels={"stack-a": {1: "PB"}},
            exclude={"stack-a": [2]},
            source_properties={"stack-a": {1: "PB", 2: "FB"}},
        )
        self.assertIs(actual_replace, replace_result)
        replace_delegate.assert_called_once_with(
            "stack-a",
            "stack-b",
            [1, "PB"],
            "merged",
            replacement_labels=[3],
            mesh_dir="custom-mesh",
            unsharded=True,
            label_names={"stack-a": {1: "PB"}},
            exclude={"stack-a": [2]},
            source_properties={"stack-a": {1: "PB", 2: "FB"}},
        )
        self.assertIs(DatastackMerger.merge, DatastackMerger.run)


if __name__ == "__main__":
    unittest.main()
