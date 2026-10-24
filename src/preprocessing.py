# Cek format WAV dan kelengkapan data.

import argparse
import hashlib
import json
from pathlib import Path
import re
import wave

import numpy as np

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
VOWELS = frozenset("aiueo")
FILENAME_PATTERN = re.compile(r"^(S[1-9][0-9]*)_([aiueo])([12])\.wav$", re.IGNORECASE)
# Open set (kelompok lain): G<kelompok>S<speaker>_<vowel>.wav, contoh G04S03_a.wav
OPEN_FILENAME_PATTERN = re.compile(r"^(G[0-9]{2}S[0-9]{2})_([aiueo])\.wav$", re.IGNORECASE)


def parse_filename(path):
    # Ambil speaker, vowel, dan nomor rekaman dari nama file.
    # File open set hanya punya satu rekaman, jadi recording = None.
    name = Path(path).name
    match = FILENAME_PATTERN.fullmatch(name)
    if match is not None:
        return {"speaker": match[1].upper(), "label": match[2].lower(),
                "recording": int(match[3])}
    match = OPEN_FILENAME_PATTERN.fullmatch(name)
    if match is not None:
        return {"speaker": match[1].upper(), "label": match[2].lower(), "recording": None}
    raise ValueError(f"Nama file tidak sesuai S<n>_<vowel><1|2>.wav "
                     f"atau G<nn>S<nn>_<vowel>.wav: {path}")


def read_validated_wav(path):
    # Baca WAV tanpa mengubah sample rate atau memotong audio.
    path = Path(path)
    if path.suffix.lower() != ".wav":
        raise ValueError(f"Bukan file WAV: {path}")
    try:
        with wave.open(str(path), "rb") as stream:
            rate = stream.getframerate()
            channels = stream.getnchannels()
            width = stream.getsampwidth()
            frames = stream.getnframes()
            if (rate, channels, width, stream.getcomptype()) != (16000, 1, 2, "NONE"):
                raise ValueError(f"Format harus PCM 16-bit mono 16000 Hz: {path} "
                                 f"(rate={rate}, channels={channels}, bits={width * 8})")
            if frames == 0:
                raise ValueError(f"Audio kosong: {path}")
            raw = stream.readframes(frames)
            if len(raw) != frames * channels * width:
                raise ValueError(f"Payload WAV terpotong/rusak: {path}")
    except (wave.Error, EOFError, OSError) as exc:
        raise ValueError(f"Tidak dapat membaca WAV {path}: {exc}") from exc
    pcm = np.frombuffer(raw, dtype="<i2")
    if not np.any(pcm):
        raise ValueError(f"Audio tanpa sinyal (semua nol): {path}")
    if np.all(pcm == pcm[0]):
        raise ValueError(f"Audio tanpa variasi sinyal (konstan): {path}")
    return pcm.astype(np.float32) / np.float32(32768.0), rate


def validate_audio_file(path):
    # Info format dan hash buat cek audio yang sama.
    audio, rate = read_validated_wav(path)
    return {"filename": Path(path).name, "sample_rate": rate, "channels": 1,
            "bit_depth": 16, "samples": int(audio.size),
            "duration_seconds": audio.size / rate,
            "signal_sha256": hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest()}


def audio_stats(audio, sample_rate):
    return {"duration_seconds": len(audio) / sample_rate,
            "rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
            "peak": float(np.max(np.abs(audio))),
            "dc_offset": float(np.mean(audio, dtype=np.float64)),
            "clipped_fraction": float(np.mean((audio >= 32767 / 32768) | (audio <= -1)))}


def preprocess_audio(audio, sample_rate, threshold_db=-30, padding_ms=80):
    # Ambang energi relatif ke bagian paling keras, bukan normalisasi volume.
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Audio harus array 1D, tidak kosong, dan berisi angka finite")
    if sample_rate <= 0 or not np.isfinite(sample_rate):
        raise ValueError("Sample rate harus positif")
    if not np.isfinite(threshold_db) or threshold_db >= 0:
        raise ValueError("threshold_db harus negatif")
    if not np.isfinite(padding_ms) or padding_ms < 0:
        raise ValueError("padding_ms tidak boleh negatif")
    before = audio_stats(audio, sample_rate)
    centered = audio - np.float32(before["dc_offset"])
    frame = max(1, round(sample_rate * 0.020))
    hop = max(1, round(sample_rate * 0.010))
    start, end = 0, len(audio)
    status = "kept_original_bounds"
    if len(audio) >= frame:
        starts = np.arange(0, len(audio), hop)
        rms = np.array([np.sqrt(np.mean(centered[s:s + frame].astype(np.float64) ** 2))
                        for s in starts])
        threshold = max(1e-4, float(rms.max()) * 10 ** (threshold_db / 20))
        active = rms >= threshold
        # Abaikan bunyi singkat di pinggir; jangan potong jeda di tengah vowel.
        edges = np.diff(np.r_[False, active, False].astype(int))
        runs = [(a, b) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
                if b - a >= 5]
        if runs:
            padding = round(sample_rate * padding_ms / 1000)
            start = max(0, int(starts[runs[0][0]]) - padding)
            end = min(len(audio), int(starts[runs[-1][1] - 1]) + frame + padding)
            status = "trimmed" if start > 0 or end < len(audio) else "kept_original_bounds"
    result = centered[start:end].astype(np.float32)
    return result, {"before": before, "after": audio_stats(result, sample_rate),
                    "start_sample": start, "end_sample": end, "trim_status": status,
                    "threshold_db": threshold_db, "padding_ms": padding_ms}


def validate_dataset(data_dir=DEFAULT_DATA_DIR, expected_speakers=None):
    # Cek jumlah, pasangan speaker-vowel, dan rekaman yang dobel.
    speakers = tuple(s.upper() for s in (
        expected_speakers if expected_speakers is not None else [f"S{i}" for i in range(1, 7)]))
    if not speakers or len(set(speakers)) != len(speakers):
        raise ValueError("expected_speakers harus berisi ID unik dan tidak kosong")
    if any(re.fullmatch(r"S[1-9][0-9]*", s) is None for s in speakers):
        raise ValueError("ID speaker harus berupa S1, S2, dan seterusnya")
    root = Path(data_dir)
    errors, records, hashes = [], [], {}
    counts, observed = {}, set()
    expected_keys = {(s, v) for s in speakers for v in VOWELS}
    for split, relative, recording in [("template", "template", 1),
                                       ("closed", "test/closed", 2)]:
        directory = root / relative
        seen = set()
        files = sorted(directory.iterdir()) if directory.is_dir() else []
        if not directory.is_dir():
            errors.append(f"Folder tidak ditemukan: {directory}")
        counts[split] = sum(p.is_file() and p.suffix.lower() == ".wav" for p in files)
        if counts[split] != len(expected_keys):
            errors.append(f"{split}: diharapkan {len(expected_keys)} WAV, ditemukan {counts[split]}")
        for path in files:
            if not path.is_file() or path.suffix.lower() != ".wav":
                errors.append(f"Entri tidak diharapkan dalam dataset: {path}")
                continue
            try:
                identity = parse_filename(path)
                key = (identity["speaker"], identity["label"])
                observed.add(identity["speaker"])
                if key in seen:
                    errors.append(f"Duplikat speaker-vowel pada {split}: {key}")
                seen.add(key)
                if identity["recording"] != recording:
                    errors.append(f"Nomor rekaman salah pada {split}: {path.name}")
                metadata = validate_audio_file(path)
                digest = metadata["signal_sha256"]
                if digest in hashes:
                    errors.append(f"Duplikat sinyal audio: {path} identik dengan {hashes[digest]}")
                else:
                    hashes[digest] = str(path)
                records.append({"split": split, **identity, **metadata})
            except ValueError as exc:
                errors.append(str(exc))
        for speaker, vowel in sorted(expected_keys - seen):
            errors.append(f"Rekaman hilang pada {split}: {speaker}_{vowel}{recording}.wav")
        for speaker, vowel in sorted(seen - expected_keys):
            errors.append(f"Speaker-vowel di luar ekspektasi pada {split}: {speaker}_{vowel}")
    return {"valid": not errors, "expected_speakers": list(speakers),
            "observed_speakers": sorted(observed), "counts": counts,
            "valid_audio_count": len(records), "errors": errors, "records": records}


def validate_open_set(data_dir=DEFAULT_DATA_DIR):
    # Cek test/open: nama file, format, tiap speaker lengkap 5 vowel, tidak ada duplikat.
    directory = Path(data_dir) / "test" / "open"
    errors, records, hashes, by_speaker = [], [], {}, {}
    if not directory.is_dir():
        return {"valid": False, "errors": [f"Folder tidak ditemukan: {directory}"],
                "speakers": [], "valid_audio_count": 0, "records": []}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".wav":
            errors.append(f"Entri tidak diharapkan dalam dataset: {path}")
            continue
        if OPEN_FILENAME_PATTERN.fullmatch(path.name) is None:
            errors.append(f"Nama file open harus G<nn>S<nn>_<vowel>.wav: {path.name}")
            continue
        try:
            identity = parse_filename(path)
            vowels = by_speaker.setdefault(identity["speaker"], [])
            if identity["label"] in vowels:
                errors.append(f"Duplikat speaker-vowel pada open: {path.name}")
            vowels.append(identity["label"])
            metadata = validate_audio_file(path)
            digest = metadata["signal_sha256"]
            if digest in hashes:
                errors.append(f"Duplikat sinyal audio: {path} identik dengan {hashes[digest]}")
            else:
                hashes[digest] = str(path)
            records.append({"split": "open", **identity, **metadata})
        except ValueError as exc:
            errors.append(str(exc))
    for speaker, vowels in sorted(by_speaker.items()):
        for vowel in sorted(VOWELS - set(vowels)):
            errors.append(f"Rekaman hilang pada open: {speaker}_{vowel}.wav")
    if not records:
        errors.append(f"Folder open kosong: {directory}")
    return {"valid": not errors, "speakers": sorted(by_speaker),
            "valid_audio_count": len(records), "errors": errors, "records": records}


def main():
    parser = argparse.ArgumentParser(description="Cek format dan kelengkapan dataset vowel")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--speakers", nargs="+", help="Default: S1 S2 S3 S4 S5 S6")
    args = parser.parse_args()
    report = validate_dataset(args.data_dir, args.speakers)
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
