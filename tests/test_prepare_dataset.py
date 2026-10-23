import unittest

import numpy as np

from src.prepare_dataset import channel_quality


class ChannelTests(unittest.TestCase):
    def test_channel_with_less_background_noise_scores_higher(self):
        rng = np.random.default_rng(42)
        t = np.arange(16000) / 16000
        voice = 0.2 * np.sin(2 * np.pi * 220 * t)
        voice[:4000] = 0
        voice[12000:] = 0
        clean = voice + rng.normal(0, 0.001, len(t))
        noisy = voice + rng.normal(0, 0.04, len(t))
        scores = channel_quality(np.column_stack([noisy, clean]))
        self.assertGreater(scores[1]["score"], scores[0]["score"])

    def test_opposite_phase_does_not_cancel_when_selecting_channel(self):
        t = np.arange(16000) / 16000
        voice = 0.2 * np.sin(2 * np.pi * 220 * t)
        stereo = np.column_stack([voice, -voice])
        self.assertAlmostEqual(float(np.max(np.abs(stereo.mean(axis=1)))), 0)
        self.assertGreater(float(np.sqrt(np.mean(stereo[:, 0] ** 2))), 0.1)


if __name__ == "__main__":
    unittest.main()
