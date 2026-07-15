import unittest

from app.ml.calibration import expected_calibration_error
from app.ml.encoders import decode_feature_attributes, decode_object_class
from app.ml.splits import audit_leakage, stratified_item_split
from app.ml.texture_features import texture_dim


class TestSplits(unittest.TestCase):
    def test_no_cross_split_leakage(self):
        rows = []
        for cls in range(3):
            for item in range(10):
                key = f"item-{cls}-{item}"
                for view in ("a", "b"):
                    rows.append({
                        "path": f"/fake/{key}_{view}.jpg",
                        "class_idx": cls,
                        "item_key": key,
                    })
        train, val, test, stats = stratified_item_split(rows, val_ratio=0.2, test_ratio=0.2, seed=42)
        report = audit_leakage(rows, val_ratio=0.2, test_ratio=0.2, seed=42)
        self.assertTrue(report.ok)
        self.assertEqual(len(train) + len(val) + len(test), len(rows))
        self.assertGreater(stats.test_items, 0)

    def test_single_item_class_goes_train(self):
        rows = [{"path": "/x/a.jpg", "class_idx": 0, "item_key": "only"}]
        train, val, test, _ = stratified_item_split(rows, seed=1)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 0)
        self.assertEqual(len(test), 0)


class TestEncoders(unittest.TestCase):
    def test_decode_object_class(self):
        import torch

        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
        label, conf = decode_object_class(logits, ["a", "b", "c", "d", "e"])
        self.assertEqual(label, "a")
        self.assertGreater(conf, 0.99)

    def test_decode_feature_attributes_respects_min_conf(self):
        import torch

        vocab = {"кельты:материал": ["камень", "металл"]}
        logits = {"кельты:материал": torch.tensor([[0.1, 0.9]])}
        lines = decode_feature_attributes(
            logits, vocab, "кельты", ["материал"], min_conf=0.95
        )
        self.assertEqual(lines, [])


class TestCalibration(unittest.TestCase):
    def test_ece_perfect(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        conf = [0.9, 0.85, 0.88, 0.92]
        report = expected_calibration_error(y_true, y_pred, conf, n_bins=5)
        self.assertLess(report.ece, 0.15)


class TestTexture(unittest.TestCase):
    def test_texture_dim(self):
        self.assertEqual(texture_dim(), 15)


if __name__ == "__main__":
    unittest.main()
