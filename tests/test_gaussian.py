# Jalankan: python -m unittest discover -s tests -v

import math
import unittest

import numpy as np

from src.data_loader import load_open_set
from src.evaluation import evaluate
from src.gaussian import Gaussian, GaussianVowelRecognizer
from src.prepare_open_set import open_name
from src.preprocessing import parse_filename, validate_open_set


def reference_nll(x, mean, cov):
    # Rumus langsung -log N(x; mean, cov) untuk pembanding.
    d = len(mean)
    diff = x - mean
    return 0.5 * (d * math.log(2 * math.pi) + math.log(np.linalg.det(cov))
                  + diff @ np.linalg.inv(cov) @ diff)


class GaussianTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(4, 4))
        self.cov = a @ a.T + 4 * np.eye(4)
        self.frames = rng.multivariate_normal([1, -2, 0.5, 3], self.cov, size=2000)

    def test_mean_and_covariance(self):
        model = Gaussian(reg=0).fit(self.frames)
        np.testing.assert_allclose(model.mean, self.frames.mean(axis=0))
        np.testing.assert_allclose(model.cov, np.cov(self.frames, rowvar=False))

    def test_nll_matches_formula(self):
        model = Gaussian(reg=0).fit(self.frames)
        x = self.frames[:5]
        expected = [reference_nll(v, model.mean, model.cov) for v in x]
        np.testing.assert_allclose(model.frame_nll(x), expected, rtol=1e-10)
        self.assertAlmostEqual(model.nll(x), float(np.mean(expected)))

    def test_diagonal_covariance(self):
        model = Gaussian("diag", reg=0).fit(self.frames)
        off_diagonal = model.cov - np.diag(np.diag(model.cov))
        self.assertTrue(np.all(off_diagonal == 0))

    def test_singular_covariance_is_regularized(self):
        frames = np.column_stack([np.arange(10.0), np.arange(10.0)])  # dua kolom identik
        model = Gaussian(reg=1e-4).fit(frames)
        self.assertTrue(np.isfinite(model.nll(frames)))

    def test_recognizer_picks_lowest_nll(self):
        rng = np.random.default_rng(1)
        centers = {"a": 0.0, "i": 5.0, "u": -5.0}
        train = [{"label": v, "mfcc": rng.normal(c, 1, size=(200, 3))} for v, c in centers.items()]
        recognizer = GaussianVowelRecognizer().fit(train)
        for vowel, center in centers.items():
            label, scores = recognizer.predict(rng.normal(center, 1, size=(50, 3)))
            self.assertEqual(label, vowel)
            self.assertEqual(label, min(scores, key=scores.get))


class EvaluationTests(unittest.TestCase):
    def test_accuracy_per_vowel_and_speaker(self):
        records = [{"filename": f"{s}_{v}", "speaker": s, "label": v}
                   for s in ["S1", "S2"] for v in "aiueo"]
        predictions = [r["label"] for r in records]
        predictions[0] = "o"  # S1 a salah
        result = evaluate(records, predictions)
        self.assertAlmostEqual(result["accuracy"], 90.0)
        self.assertAlmostEqual(result["per_vowel"]["a"]["accuracy"], 50.0)
        self.assertAlmostEqual(result["per_speaker"]["S1"]["accuracy"], 80.0)
        self.assertEqual(result["confusion"]["a"]["o"], 1)


class OpenSetTests(unittest.TestCase):
    def test_raw_names_are_normalized(self):
        self.assertEqual(open_name("G04_S03_A.wav"), "G04S03_a.wav")
        self.assertEqual(open_name("G09_S01_E01.wav"), "G09S01_e.wav")
        self.assertEqual(open_name("Kel3_S05_U.wav"), "G03S05_u.wav")
        with self.assertRaises(ValueError):
            open_name("S1_a1.wav")

    def test_open_filename_identity(self):
        self.assertEqual(parse_filename("G04S03_a.wav"),
                         {"speaker": "G04S03", "label": "a", "recording": None})

    def test_actual_open_set(self):
        report = validate_open_set()
        self.assertTrue(report["valid"], report["errors"])
        records = load_open_set()
        self.assertEqual(len(records), 5 * len(report["speakers"]))
        self.assertTrue(all(r["sample_rate"] == 16000 for r in records))


if __name__ == "__main__":
    unittest.main()
