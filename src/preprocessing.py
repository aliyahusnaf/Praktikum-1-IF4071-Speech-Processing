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


def parse_filename(path):
    # Ambil speaker, vowel, dan nomor rekaman dari nama file.
    match = FILENAME_PATTERN.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"Nama file tidak sesuai S<n>_<vowel><1|2>.wav: {path}")
    return {"speaker": match[1].upper(), "label": match[2].lower(),
            "recording": int(match[3])}


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
