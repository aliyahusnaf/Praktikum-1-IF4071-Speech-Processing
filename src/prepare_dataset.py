# Buat ulang WAV dari raw, jadi tidak menumpuk trimming sebelumnya.

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import wave
import zipfile
from datetime import datetime

import numpy as np

from .preprocessing import DEFAULT_DATA_DIR, preprocess_audio, validate_dataset

SPEAKERS = {"aliya": "S1", "farrel": "S2", "fithra": "S3",
            "kefas": "S4", "rakdaf": "S5", "timur": "S6"}


def decode_raw(path, ffmpeg):
    result = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", "pipe:1"],
        capture_output=True, check=True)
    with wave.open(io.BytesIO(result.stdout), "rb") as stream:
        channels = stream.getnchannels()
        pcm = stream.readframes(stream.getnframes())
    audio = np.frombuffer(pcm, dtype="<i2").reshape(-1, channels).astype(np.float32) / 32768
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"Audio raw tidak terbaca: {path}")
    return audio, result.stderr.decode(errors="replace").strip()


def channel_quality(audio):
    scores = []
    for channel in audio.T:
        centered = channel.astype(np.float64) - channel.mean(dtype=np.float64)
        frames = [centered[s:s + 320] for s in range(0, len(centered), 160)]
        rms = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
        contrast = 20 * np.log10(max(float(np.percentile(rms, 90)), 1e-8) /
                                 max(float(np.percentile(rms, 20)), 1e-8))
        clipping = float(np.mean((channel >= 32767 / 32768) | (channel <= -1)))
        scores.append({"energy_contrast_db": float(contrast), "clipped_fraction": clipping,
                       "score": float(contrast - 1000 * clipping)})
    return scores


def prepare_dataset(ffmpeg):
    root = DEFAULT_DATA_DIR.parents[1]
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    sources = []
    for name, speaker in SPEAKERS.items():
        for vowel in "aiueo":
            for recording in [1, 2]:
                stem = f"{vowel}{recording}"
                matches = [p for p in (root / "data/raw" / name).iterdir()
                           if p.is_file() and p.stem == stem and p.suffix.lower() in {".wav", ".m4a"}]
                if len(matches) != 1:
                    raise ValueError(f"Raw harus tepat satu untuk {name}/{stem}: {matches}")
                sources.append((matches[0], speaker, stem))
    raw_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p, _, _ in sources}
    decoded, choices = [], {}
    for path, speaker, stem in sources:
        audio, warning = decode_raw(path, ffmpeg)
        quality = channel_quality(audio)
        decoded.append((path, speaker, stem, audio, warning, quality))
    # Pilih channel yang sama untuk satu speaker berdasarkan median kontras energi.
    for speaker in SPEAKERS.values():
        group = [r for r in decoded if r[1] == speaker]
        counts = {r[3].shape[1] for r in group}
        if len(counts) != 1:
            raise ValueError(f"Jumlah channel raw tidak konsisten: {speaker}")
        scores = np.array([[q["score"] for q in r[5]] for r in group])
        choices[speaker] = int(np.argmax(np.median(scores, axis=0)))
    rows = []
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        stage = Path(tmp)
        for path, speaker, stem, audio, warning, quality in decoded:
            channel = choices[speaker]
            trimmed, info = preprocess_audio(audio[:, channel], 16000)
            pcm = np.clip(np.rint(trimmed.astype(np.float64) * 32768), -32768, 32767).astype("<i2")
            relative = Path("template" if stem.endswith("1") else "test/closed") / f"{speaker}_{stem}.wav"
            dest = stage / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(dest), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(pcm.tobytes())
            rows.append({"filename": dest.name, "source": str(path.relative_to(root)),
                         "source_sha256": raw_hashes[path], "source_channels": audio.shape[1],
                         "selected_channel": channel, "channel_quality": quality,
                         "decoder_messages": warning, **info})
        report = validate_dataset(stage)
        if not report["valid"]:
            raise ValueError(report["errors"])
        backup = output / f"processed_before_channel_fix_{datetime.now():%Y%m%d_%H%M%S_%f}.zip"
        with zipfile.ZipFile(backup, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in DEFAULT_DATA_DIR.rglob("*.wav"):
                archive.write(path, path.relative_to(root))
        for path in stage.rglob("*.wav"):
            path.replace(DEFAULT_DATA_DIR / path.relative_to(stage))
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in raw_hashes.items())
    (output / "preprocessing_report.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output / "validation_closed.json").write_text(json.dumps(validate_dataset(), indent=2), encoding="utf-8")
    print("Channel terpilih (0=kiri, 1=kanan):", choices)
    print("Selesai:", len(rows), "WAV; raw tetap utuh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Buat ulang processed dari raw")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    prepare_dataset(parser.parse_args().ffmpeg)
