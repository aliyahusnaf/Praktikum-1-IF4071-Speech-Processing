# Siapkan data uji open speaker dari rekaman kelompok lain.
# Sumber: data/raw/open/<A|E|I|O|U>/<kelompok>_S<nn>_<vowel>[nn].wav
# Hasil:  data/processed/test/open/G<nn>S<nn>_<vowel>.wav

import argparse
import hashlib
import json
from pathlib import Path
import re
import wave

import numpy as np

from .preprocessing import DEFAULT_DATA_DIR, preprocess_audio, read_validated_wav, validate_open_set

ROOT = DEFAULT_DATA_DIR.parents[1]
OPEN_RAW_DIR = ROOT / "data" / "raw" / "open"
# Contoh: G04_S03_A.wav, G09_S01_A01.wav, Kel3_S01_A.wav
RAW_PATTERN = re.compile(r"^(?:G|Kel)0*([0-9]+)_S0*([0-9]+)_([AIUEO])[0-9]*\.wav$", re.IGNORECASE)


def open_name(raw_name):
    # Nama baru yang seragam: kelompok dan speaker dua digit.
    match = RAW_PATTERN.fullmatch(raw_name)
    if match is None:
        raise ValueError(f"Nama file open tidak dikenali: {raw_name}")
    return f"G{int(match[1]):02d}S{int(match[2]):02d}_{match[3].lower()}.wav"


def write_wav(path, audio, rate=16000):
    pcm = np.clip(np.rint(audio.astype(np.float64) * 32768), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm.tobytes())


def prepare_open_set(raw_dir=OPEN_RAW_DIR, data_dir=DEFAULT_DATA_DIR):
    raw_dir, out_dir = Path(raw_dir), Path(data_dir) / "test" / "open"
    sources = sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    if not sources:
        raise ValueError(f"Tidak ada WAV di {raw_dir}")
    names = {}
    for path in sources:
        name = open_name(path.name)
        if name in names:
            raise ValueError(f"Dua file menjadi {name}: {names[name]} dan {path}")
        # Folder vowel harus sama dengan vowel di nama file.
        if path.parent.name.lower() != name[-5]:
            raise ValueError(f"Vowel folder dan nama file berbeda: {path}")
        names[name] = path
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.wav"):
        old.unlink()
    rows = []
    for name, path in sorted(names.items()):
        # Sama dengan data closed: hapus DC offset dan trimming jeda awal/akhir.
        audio, rate = read_validated_wav(path)
        trimmed, info = preprocess_audio(audio, rate)
        write_wav(out_dir / name, trimmed, rate)
        rows.append({"filename": name, "source": path.relative_to(ROOT).as_posix(),
                     "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **info})
    report = validate_open_set(data_dir)
    if not report["valid"]:
        raise ValueError(report["errors"])
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    (output / "preprocessing_open_report.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Selesai: {len(rows)} WAV dari {len(report['speakers'])} speaker -> {out_dir}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Siapkan data uji open speaker")
    parser.add_argument("--raw-dir", type=Path, default=OPEN_RAW_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    prepare_open_set(args.raw_dir, args.data_dir)
