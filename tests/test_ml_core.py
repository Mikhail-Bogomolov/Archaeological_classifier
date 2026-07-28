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


class TestItemFailures(unittest.TestCase):
    def test_summarize_item_failures_flags_all_wrong(self):
        from app.ml.evaluate_classifier import _summarize_item_failures

        by_item = {
            "ok-item": {
                "true_idx": 0,
                "preds": [0, 0],
                "oks": [True, True],
                "paths": ["a.jpg", "b.jpg"],
                "confs": [0.9, 0.8],
            },
            "bad-item": {
                "true_idx": 3,
                "preds": [1, 1, 1],
                "oks": [False, False, False],
                "paths": ["x_a.jpg", "x_b.jpg", "x_v.jpg"],
                "confs": [0.9, 0.85, 0.8],
            },
        }
        correct, total, failed = _summarize_item_failures(by_item)
        self.assertEqual(correct, 1)
        self.assertEqual(total, 2)
        self.assertEqual(len(failed), 1)
        self.assertTrue(failed[0]["all_wrong"])
        self.assertEqual(failed[0]["item_key"], "bad-item")
        self.assertEqual(failed[0]["majority_wrong_pred"], "ножи")


class TestSohrannostNormalize(unittest.TestCase):
    def test_typo_and_free_text_collapse(self):
        from app.ml.table_normalization import coarse_sohrannost, normalize_cell_value

        self.assertEqual(
            coarse_sohrannost("целый, на одной сторон отверстие (недолив)"),
            "целый",
        )
        self.assertEqual(coarse_sohrannost("целые"), "целый")
        self.assertEqual(coarse_sohrannost("обломки"), "сломан")
        self.assertEqual(coarse_sohrannost("целые и сломаны"), "сломан")
        self.assertEqual(coarse_sohrannost("Фрагмент"), "сломан")
        self.assertEqual(coarse_sohrannost("половина"), "сломан")
        self.assertEqual(
            normalize_cell_value("кельты", "сохранность", "Целая"),
            "целый",
        )

    def test_coarse_weak_heads(self):
        from app.ml.table_normalization import (
            coarse_kreplenie,
            coarse_nakladka_forma,
            coarse_tip_okonchania,
            coarse_knife_tip,
            coarse_knife_rukoyat,
            normalize_cell_value,
        )

        self.assertEqual(
            coarse_tip_okonchania("подтреугольно-кольчатое с округлым отверстием"),
            "подтреугольное",
        )
        self.assertEqual(coarse_tip_okonchania("овально-кольчатое"), "кольчато-овальное")
        self.assertEqual(coarse_tip_okonchania("стремечковидные"), "кольчато-овальное")
        self.assertEqual(coarse_tip_okonchania("трапециевидное кольчатое"), "подтреугольное")
        self.assertEqual(coarse_kreplenie("шпенек"), "внутреннее")
        self.assertEqual(coarse_kreplenie("петелька"), "внешнее")
        self.assertEqual(coarse_nakladka_forma("круглая с зубчатым краем"), "прямоугольная")
        self.assertIsNone(coarse_nakladka_forma("ажурная"))
        self.assertEqual(coarse_knife_tip("выпуклообушковый"), "изогнутый")
        self.assertEqual(coarse_knife_tip("прямолезвийный"), "прямой")
        self.assertEqual(coarse_knife_rukoyat("петельчатая рукоять"), "выделенная")
        self.assertEqual(
            normalize_cell_value("удила", "тип_окончания", "кольчатое"),
            "кольчато-овальное",
        )
        self.assertIsNone(
            normalize_cell_value(
                "ножи",
                "материал",
                "Железо, латунь, медь, алюминий, свинец, пластмасса",
            )
        )

    def test_vocab_includes_knife_rukoyat(self):
        from app.ml.feature_vocab import build_vocab

        vocab = build_vocab()
        self.assertIn("ножи:рукоять", vocab)
        self.assertEqual(
            set(vocab["ножи:рукоять"]),
            {"выделенная", "невыделенная"},
        )


class TestTexture(unittest.TestCase):
    def test_texture_dim(self):
        self.assertEqual(texture_dim(), 17)

    def test_geometry_appended(self):
        from PIL import Image

        from app.ml.texture_features import extract_texture_vector

        img = Image.new("RGB", (200, 80), color=(200, 200, 200))
        # тёмный вытянутый «предмет» по центру
        for x in range(40, 160):
            for y in range(30, 50):
                img.putpixel((x, y), (40, 40, 40))
        vec = extract_texture_vector(img)
        self.assertEqual(len(vec), 17)
        self.assertGreater(float(vec[-2]), 0.2)  # aspect_n
        self.assertGreater(float(vec[-1]), 0.0)  # solidity


class TestConfusedItem(unittest.TestCase):
    def test_14_11_flagged(self):
        from app.ml.table_normalization import is_confused_item

        self.assertTrue(is_confused_item("уд85_14-11_а.jpg"))
        self.assertFalse(is_confused_item("уд85_9-1_а.jpg"))


if __name__ == "__main__":
    unittest.main()
