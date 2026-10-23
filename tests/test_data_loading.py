# Jalankan: python -m unittest discover -s tests -v

from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from src.data_loader import load_audio, load_dataset
from src.preprocessing import (DEFAULT_DATA_DIR, parse_filename,
                               read_validated_wav, validate_dataset, preprocess_audio)

SPEAKERS = [f"S{i}" for i in range(1, 7)]


def write_wav(path, samples, rate=16000, channels=1, width=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(width)
        stream.setframerate(rate)
        stream.writeframes(np.asarray(samples, dtype="<i2").tobytes())


class ActualDatasetTests(unittest.TestCase):
    def test_actual_dataset_and_mfcc_interface(self):
        report = validate_dataset()
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["valid_audio_count"], 60)
        data = load_dataset()
        self.assertEqual({k: len(v) for k, v in data.items()}, {"template": 30, "closed": 30})
        for split in data.values():
            self.assertEqual({(r["speaker"], r["label"]) for r in split},
                             {(s, v) for s in SPEAKERS for v in "aiueo"})
            for record in split:
                self.assertEqual(record["audio"].dtype, np.float32)
                self.assertEqual(record["audio"].ndim, 1)
                self.assertEqual(record["sample_rate"], 16000)
                self.assertTrue(np.isfinite(record["audio"]).all())
                self.assertGreater(record["audio"].size, 0)
                raw = load_audio(record["path"], preprocess=False)["audio"]
                self.assertIsNone(record["preprocessing"])
                np.testing.assert_array_equal(record["audio"], raw)

    def test_five_speaker_expectation_reports_mismatch(self):
        report = validate_dataset(expected_speakers=SPEAKERS[:5])
        self.assertFalse(report["valid"])
        self.assertEqual(report["counts"], {"template": 30, "closed": 30})
        self.assertTrue(any("S6" in e for e in report["errors"]))
        with self.assertRaises(ValueError):
            load_dataset(expected_speakers=SPEAKERS[:5])

    def test_single_file_identity(self):
        record = load_audio(DEFAULT_DATA_DIR / "template" / "S2_a1.wav")
        self.assertEqual((record["speaker"], record["label"], record["recording"]), ("S2", "a", 1))


class InvalidInputTests(unittest.TestCase):
    def test_invalid_wavs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "S1_a1.wav"
            cases = [([], 16000, 1, 2), ([0, 0], 16000, 1, 2),
                     ([7, 7], 16000, 1, 2), ([1, 2], 8000, 1, 2),
                     ([1, 2], 16000, 2, 2), ([1, 2], 16000, 1, 1)]
            for samples, rate, channels, width in cases:
                with self.subTest(rate=rate, channels=channels, width=width, samples=samples):
                    write_wav(path, samples, rate, channels, width)
                    with self.assertRaises(ValueError):
                        read_validated_wav(path)
            path.write_bytes(b"not a WAV")
            with self.assertRaises(ValueError):
                read_validated_wav(path)
            write_wav(path, [0, 1, 2, 3])
            path.write_bytes(path.read_bytes()[:-2])
            with self.assertRaisesRegex(ValueError, "terpotong"):
                read_validated_wav(path)

    def test_pcm_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "S1_a1.wav"
            write_wav(path, [-32768, 0, 32767])
            audio, _ = read_validated_wav(path)
            np.testing.assert_array_equal(audio, np.array([-1, 0, 32767 / 32768], dtype=np.float32))

    def test_filename(self):
        for name in ["a1.wav", "S1_x1.wav", "S1_a3.wav"]:
            with self.assertRaises(ValueError):
                parse_filename(name)

    def test_missing_duplicate_and_wrong_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for vowel in "aiueo":
                for recording, folder in [(1, "template"), (2, "test/closed")]:
                    # Bikin sinyal yang beda untuk tiap rekaman.
                    write_wav(root / folder / f"S1_{vowel}{recording}.wav",
                              [0, ord(vowel), recording])
            self.assertTrue(validate_dataset(root, ["S1"])["valid"])
            loaded = load_dataset(root, ["S1"])
            self.assertEqual([len(loaded[k]) for k in ["template", "closed"]], [5, 5])
            (root / "template/S1_a1.wav").unlink()
            report = validate_dataset(root, ["S1"])
            self.assertTrue(any("Rekaman hilang" in e for e in report["errors"]))
            write_wav(root / "template/S1_a1.wav", [0, ord("a"), 2])
            write_wav(root / "template/S1_a2.wav", [0, 42, 3])
            report = validate_dataset(root, ["S1"])
            for message in ["Duplikat sinyal", "Duplikat speaker-vowel", "Nomor rekaman salah"]:
                self.assertTrue(any(message in e for e in report["errors"]), report)


class PreprocessingTests(unittest.TestCase):
    def test_trim_keeps_padding_and_middle_silence(self):
        rate = 16000
        t = np.arange(rate // 2) / rate
        vowel = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        audio = np.concatenate([np.zeros(8000), vowel, np.zeros(3200), vowel, np.zeros(8000)]).astype(np.float32)
        original = audio.copy()
        result, info = preprocess_audio(audio, rate)
        np.testing.assert_array_equal(audio, original)
        self.assertLessEqual(info["start_sample"], 8000 - 1280)
        self.assertGreaterEqual(info["end_sample"], len(audio) - 8000 + 1280)
        self.assertGreater(info["start_sample"], 0)
        self.assertLess(info["end_sample"], len(audio))
        np.testing.assert_allclose(result, audio[info["start_sample"]:info["end_sample"]], atol=1e-7)

    def test_dc_offset_and_short_recording(self):
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result, info = preprocess_audio(audio, 16000)
        self.assertEqual(len(result), len(audio))
        self.assertAlmostEqual(float(result.mean()), 0, places=6)
        self.assertEqual(info["trim_status"], "kept_original_bounds")

    def test_no_activity_keeps_bounds_and_invalid_input_rejected(self):
        audio = np.full(16000, 1e-6, dtype=np.float32)
        result, info = preprocess_audio(audio, 16000)
        self.assertEqual(len(result), len(audio))
        for bad in [[], [[1, 2]], [np.nan], [np.inf]]:
            with self.assertRaises(ValueError):
                preprocess_audio(bad, 16000)


if __name__ == "__main__":
    unittest.main()
