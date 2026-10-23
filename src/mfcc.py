# ekstraksi MFCC 39 dimensi per frame dengan python_speech_features

import argparse
import json
import math
from pathlib import Path

import numpy as np
from python_speech_features import delta, mfcc

from .data_loader import load_dataset
from .preprocessing import DEFAULT_DATA_DIR

# parameter
WINLEN = 0.025     # lebar frame 25 ms
WINSTEP = 0.010    # geser 10 ms (overlap 15 ms)
PREEMPH = 0.97     # koefisien pre-emphasis
NFILT = 26         # jumlah filter mel
NUMCEP = 13        # c0 (diganti log energi) + c1..c12
DELTA_N = 2        # delta dari 2 frame maju/mundur
EPS = 1e-10

FEATURE_NAMES = ([f"{p}{n}" for p in ["", "d_", "dd_"]
                  for n in ["energy"] + [f"c{i}" for i in range(1, 13)]])


def _nfft(sample_rate):
    # minimal 512 dan tidak lebih pendek dari frame
    return max(512, 2 ** math.ceil(math.log2(round(WINLEN * sample_rate))))


def extract_mfcc_39(audio, sample_rate):
    # hasil (jumlah_frame, 39) = [energi, c1..c12, delta, delta-delta]
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Audio harus array 1D, tidak kosong, dan berisi angka finite")
    if sample_rate <= 0 or not np.isfinite(sample_rate):
        raise ValueError("Sample rate harus positif")
    # c0 diganti log energi, lifter dimatikan karena tidak ada di materi
    static = mfcc(audio, samplerate=sample_rate, winlen=WINLEN, winstep=WINSTEP,
                  numcep=NUMCEP, nfilt=NFILT, nfft=_nfft(sample_rate), preemph=PREEMPH,
                  ceplifter=0, appendEnergy=True, winfunc=np.hamming)
    d1 = delta(static, DELTA_N)
    d2 = delta(d1, DELTA_N)
    return np.hstack([static, d1, d2])


def apply_cmvn(features, stats_from=None):
    # mean 0 dan std 1 tiap dimensi, statistik dari stats_from (jika diisi)
    features = np.asarray(features, dtype=np.float64)
    ref = features if stats_from is None else np.asarray(stats_from, dtype=np.float64)
    std = ref.std(axis=0)
    return (features - ref.mean(axis=0)) / np.where(std < EPS, 1.0, std)


def extract_features(records, cmvn=False):
    # tambah key "mfcc" ke tiap record dari load_dataset()
    # cmvn per speaker
    result = [{**r, "mfcc": extract_mfcc_39(r["audio"], r["sample_rate"])} for r in records]
    if cmvn:
        pooled = {}
        for r in result:
            pooled.setdefault(r["speaker"], []).append(r["mfcc"])
        pooled = {s: np.vstack(f) for s, f in pooled.items()}
        result = [{**r, "mfcc": apply_cmvn(r["mfcc"], pooled[r["speaker"]])} for r in result]
    return result


def main():
    parser = argparse.ArgumentParser(description="Ekstraksi MFCC 39 dimensi untuk dataset vowel")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cmvn", action="store_true", help="Normalisasi CMVN per speaker")
    parser.add_argument("--output", type=Path, help="Simpan fitur ke file .npz (key = nama file)")
    args = parser.parse_args()
    data = load_dataset(args.data_dir)
    summary, arrays = {}, {}
    for split, records in data.items():
        features = extract_features(records, args.cmvn)
        summary[split] = {r["filename"]: list(r["mfcc"].shape) for r in features}
        arrays.update({f"{split}/{r['filename']}": r["mfcc"] for r in features})
    print(json.dumps(summary, indent=2))
    if args.output:
        np.savez(args.output, **arrays)
        print(f"Fitur disimpan ke {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
