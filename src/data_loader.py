# Baca audio untuk input MFCC.

from pathlib import Path

from .preprocessing import (DEFAULT_DATA_DIR, parse_filename, read_validated_wav,
                            validate_dataset, validate_open_set, preprocess_audio)


def load_audio(path, preprocess=False):
    # Baca satu file beserta labelnya.
    path = Path(path)
    identity = parse_filename(path)
    audio, rate = read_validated_wav(path)
    processing = None
    if preprocess:
        audio, processing = preprocess_audio(audio, rate)
    return {"audio": audio, "sample_rate": rate, **identity,
            "filename": path.name, "path": str(path.resolve()), "preprocessing": processing}


def load_dataset(data_dir=DEFAULT_DATA_DIR, expected_speakers=None, preprocess=False):
    # Hasilnya dua list: template dan closed.
    report = validate_dataset(data_dir, expected_speakers)
    if not report["valid"]:
        raise ValueError("Dataset tidak valid:\n- " + "\n- ".join(report["errors"]))
    root = Path(data_dir)
    result = {}
    for split, relative in [("template", "template"), ("closed", "test/closed")]:
        result[split] = [load_audio(path, preprocess=preprocess)
                         for path in sorted((root / relative).iterdir())]
    return result


def load_open_set(data_dir=DEFAULT_DATA_DIR):
    # Data uji open speaker dari kelompok lain (data/processed/test/open).
    report = validate_open_set(data_dir)
    if not report["valid"]:
        raise ValueError("Open set tidak valid:\n- " + "\n- ".join(report["errors"]))
    directory = Path(data_dir) / "test" / "open"
    return [load_audio(path) for path in sorted(directory.iterdir())]
