"""Regression coverage for the legacy image-only entrypoint (issue #2).

Run from the repository root with:
    python -m unittest discover -s tests -p test_legacy_inference.py
"""

from collections import Counter
import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class EntrypointTests(unittest.TestCase):
    def test_help_needs_no_third_party_packages(self):
        # -S disables site-packages, where torch and the training dependencies live.
        with tempfile.TemporaryDirectory() as working_dir:
            result = subprocess.run(
                [sys.executable, "-S", str(REPO_ROOT / "test.py"), "--help"],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=15,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output_dir", result.stdout)
        self.assertIn("test_image", result.stdout)


class ImageInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep the help test independently runnable without the inference stack.
        cls.np = importlib.import_module("numpy")
        cls.torch = importlib.import_module("torch")
        cls.Image = importlib.import_module("PIL.Image")
        cls.ImageDataset = importlib.import_module("lib.dataset").ImageDataset
        cls.utility = importlib.import_module("lib.utility")
        cls.Alignment = importlib.import_module("conf.alignment").Alignment
        cls.tester = importlib.import_module("tester")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def image(self, relative, color=(20, 80, 160)):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        self.Image.new("RGB", (19, 13), color).save(path)
        return path

    def dataset(self, source):
        return self.ImageDataset(
            "",
            image_dir=str(source),
            width=256,
            height=256,
            classes_num=[68, 9, 68],
            crop_op=False,
            aug_prob=0,
            edge_info=[],
            flip_mapping=[],
            is_train=False,
        )

    def test_single_jpeg_and_png_accept_file_paths(self):
        for name in ("face.jpg", "face.PNG"):
            with self.subTest(name=name):
                path = self.image(name)
                original = path.read_bytes()
                dataset = self.dataset(path)
                self.assertEqual(len(dataset), 1)
                sample = dataset[0]
                self.assertEqual(Path(sample["image_path"]), path.resolve())
                self.assertEqual(tuple(sample["data"].shape), (3, 256, 256))
                self.assertTrue(self.torch.isfinite(sample["data"]).all().item())
                self.assertEqual(path.read_bytes(), original)

    def test_directory_includes_nested_images_and_ignores_other_files(self):
        paths = [
            self.image("inputs/face.jpg"),
            self.image("inputs/nested/face.PNG"),
        ]
        (self.root / "inputs" / "notes.txt").write_text("not an image")
        dataset = self.dataset(self.root / "inputs")
        self.assertEqual(
            dataset.image_paths,
            sorted(str(path.resolve()) for path in paths),
        )
        self.assertEqual(len(dataset), 2)

    def test_missing_empty_and_unsupported_inputs_fail_clearly(self):
        with self.assertRaises(FileNotFoundError):
            self.dataset(self.root / "missing.jpg")
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(ValueError):
            self.dataset(empty)
        unsupported = self.root / "notes.txt"
        unsupported.write_text("not an image")
        with self.assertRaises(ValueError):
            self.dataset(unsupported)

    def test_corrupt_image_fails_when_read(self):
        path = self.root / "broken.png"
        path.write_bytes(b"not a PNG image")
        dataset = self.dataset(path)
        with self.assertRaises(ValueError):
            dataset[0]

    def test_image_mode_preserves_explicit_file_and_output_directory(self):
        path = self.image("face.jpg")
        output_dir = self.root / "results"
        config = self.Alignment(SimpleNamespace(
            mode="test_image",
            image_dir=str(path),
            output_dir=str(output_dir),
            data_definition="300W",
        ))
        self.assertEqual(Path(config.test_pic_dir), path)
        self.assertEqual(Path(config.output_dir), output_dir)

    def test_legacy_checkpoint_metadata_and_strict_cpu_loading(self):
        # Original published .pkl files include training metadata in addition to
        # weights. Counter metadata reproduces the weights_only loading failure.
        original = self.torch.nn.Linear(3, 2)
        with self.torch.no_grad():
            original.weight.fill_(0.25)
            original.bias.fill_(-0.5)
        checkpoint_path = self.root / "legacy.pkl"
        self.torch.save({
            "net": original.state_dict(),
            "training_metadata": Counter({"epochs": 300}),
        }, checkpoint_path)
        config = SimpleNamespace(device=self.torch.device("cpu"), model_dir=str(self.root))
        args = SimpleNamespace(pretrained_weight=str(checkpoint_path))
        restored = self.torch.nn.Linear(3, 2)
        self.tester._load_checkpoint(restored, config, args)
        for name, tensor in restored.state_dict().items():
            self.assertEqual(tensor.device.type, "cpu")
            self.torch.testing.assert_close(tensor, original.state_dict()[name])
        # A mismatched network must still be rejected instead of silently
        # dropping checkpoint weights while restoring legacy loading support.
        with self.assertRaises(RuntimeError):
            self.tester._load_checkpoint(self.torch.nn.Linear(4, 2), config, args)

    def test_cpu_forward_saves_every_batch_item_without_touching_inputs(self):
        paths = [
            self.image("inputs/face.jpg"),
            self.image("inputs/nested/face.jpg", color=(10, 40, 100)),
            self.image("inputs/nested/face.png", color=(70, 90, 120)),
        ]
        before = {path: path.read_bytes() for path in paths}
        source_dir = self.root / "inputs"
        dataset = self.dataset(source_dir)
        loader = self.torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        torch = self.torch

        class SmallNetwork(torch.nn.Module):
            def forward(self, images):
                points = torch.zeros(
                    (images.shape[0], 68, 2), dtype=images.dtype, device=images.device
                )
                return [], [], points

        output_dir = self.root / "results" / "nested-output"
        config = SimpleNamespace(
            device=torch.device("cpu"),
            output_dir=str(output_dir),
            ckpt_dir=str(self.root / "checkpoints"),
            data_definition="regression",
            width=256,
            height=256,
        )
        results = self.utility.forward_test_image_only(config, loader, SmallNetwork())
        self.assertEqual(len(results), len(paths))
        self.assertEqual(
            {Path(result["image_path"]) for result in results},
            {path.resolve() for path in paths},
        )
        for result in results:
            input_path = Path(result["image_path"])
            relative = input_path.relative_to(source_dir.resolve())
            visual = Path(result["visualization"])
            landmarks = Path(result["landmarks"])
            self.assertEqual(visual, output_dir.resolve() / relative)
            self.assertEqual(landmarks, output_dir.resolve() / (str(relative) + ".npy"))
            self.assertTrue(visual.is_file())
            self.assertTrue(landmarks.is_file())
            points = self.np.load(landmarks, allow_pickle=False)
            self.assertEqual(points.shape, (68, 2))
            self.assertTrue(self.np.isfinite(points).all())
        self.assertEqual(len(list(output_dir.rglob("*.npy"))), len(paths))
        self.assertEqual(list(source_dir.rglob("*.npy")), [])
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
