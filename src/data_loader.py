# Baca audio untuk input MFCC.

from pathlib import Path

from .preprocessing import (DEFAULT_DATA_DIR, parse_filename,
                            read_validated_wav, validate_dataset)


def load_audio(path):
    # Baca satu file beserta labelnya.
    path = Path(path)
    identity = parse_filename(path)
    audio, rate = read_validated_wav(path)
    return {"audio": audio, "sample_rate": rate, **identity,
            "filename": path.name, "path": str(path.resolve())}


def load_dataset(data_dir=DEFAULT_DATA_DIR, expected_speakers=None):
    # Hasilnya dua list: template dan closed.
    report = validate_dataset(data_dir, expected_speakers)
    if not report["valid"]:
        raise ValueError("Dataset tidak valid:\n- " + "\n- ".join(report["errors"]))
    root = Path(data_dir)
    result = {}
    for split, relative in [("template", "template"), ("closed", "test/closed")]:
        result[split] = [load_audio(path) for path in sorted((root / relative).iterdir())]
    return result
