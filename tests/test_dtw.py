import unittest

import numpy as np

from src.dtw import DTWVowelRecognizer, average_template, dtw, local_cost, medoid


class DTWTests(unittest.TestCase):
    def test_local_cost_is_euclidean(self):
        x = np.array([[0.0, 0.0], [3.0, 4.0]])
        y = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 0.0]])
        expected = [[np.linalg.norm(a - b) for b in y] for a in x]
        np.testing.assert_allclose(local_cost(x, y), expected, atol=1e-12)

    def test_identical_sequences_cost_zero(self):
        x = np.random.default_rng(0).normal(size=(20, 39))
        distance, path = dtw(x, x, return_path=True)
        self.assertAlmostEqual(distance, 0.0)
        self.assertEqual(path, [(i, i) for i in range(20)])

    def test_manual_example(self):
        # Path terbaik (0,0) -> (0,1) -> (1,2): 2*0 + 1 + 2*0 = 1, dibagi N + M = 5.
        x = np.array([[1.0], [3.0]])
        y = np.array([[1.0], [2.0], [3.0]])
        distance, path = dtw(x, y, return_path=True)
        self.assertAlmostEqual(distance, 1.0 / 5.0)
        self.assertEqual(path, [(0, 0), (0, 1), (1, 2)])

    def test_time_stretched_sequence_cost_zero(self):
        x = np.random.default_rng(1).normal(size=(15, 5))
        stretched = np.repeat(x, 2, axis=0)
        self.assertAlmostEqual(dtw(x, stretched), 0.0)

    def test_path_is_monotonic_and_complete(self):
        rng = np.random.default_rng(2)
        _, path = dtw(rng.normal(size=(12, 3)), rng.normal(size=(17, 3)), return_path=True)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (11, 16))
        for (i0, j0), (i1, j1) in zip(path, path[1:]):
            self.assertIn((i1 - i0, j1 - j0), [(1, 1), (1, 0), (0, 1)])

    def test_medoid_and_average_of_identical_sequences(self):
        x = np.random.default_rng(3).normal(size=(10, 4))
        self.assertEqual(medoid([x + 3.0, x, x - 3.0]), 1)
        np.testing.assert_allclose(average_template([x, x, x], iterations=2), x)

    def test_recognizer_picks_lowest_cost(self):
        rng = np.random.default_rng(4)
        centers = {"a": 0.0, "i": 5.0, "u": -5.0}
        train = [{"label": v, "mfcc": rng.normal(c, 1, size=(int(rng.integers(20, 40)), 3))}
                 for v, c in centers.items() for _ in range(3)]
        recognizer = DTWVowelRecognizer(iterations=2).fit(train)
        for vowel, center in centers.items():
            label, scores = recognizer.predict(rng.normal(center, 1, size=(25, 3)))
            self.assertEqual(label, vowel)
            self.assertEqual(label, min(scores, key=scores.get))


if __name__ == "__main__":
    unittest.main()
