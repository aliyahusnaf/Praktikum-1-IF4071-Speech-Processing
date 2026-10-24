# Sistem 2: pengenalan vowel dengan Model Gaussian (mean + covariance).
#
# Latih: semua frame MFCC 39 dimensi dari data template satu vowel digabung,
#        lalu dihitung mean (39) dan covariance (39 x 39) -> satu Gaussian per vowel.
# Uji:   audio -> MFCC -> negative log-likelihood (NLL) terhadap tiap model
#        -> pilih NLL terkecil -> hasil vowel.

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .data_loader import load_dataset, load_open_set
from .evaluation import VOWEL_ORDER, evaluate, format_confusion
from .mfcc import extract_features
from .preprocessing import DEFAULT_DATA_DIR

LOG_2PI = math.log(2 * math.pi)
DEFAULT_REG = 1e-4  # tambahan kecil di diagonal supaya covariance pasti bisa diinvers


class Gaussian:
    # Distribusi Gaussian multivariat N(mean, covariance) untuk satu vowel.

    def __init__(self, covariance="full", reg=DEFAULT_REG):
        if covariance not in ("full", "diag"):
            raise ValueError("covariance harus 'full' atau 'diag'")
        self.covariance_type = covariance
        self.reg = reg

    def fit(self, frames):
        frames = np.asarray(frames, dtype=np.float64)
        n, dim = frames.shape
        if n < 2:
            raise ValueError("Butuh minimal 2 frame untuk menghitung covariance")
        self.mean = frames.mean(axis=0)
        centered = frames - self.mean
        cov = centered.T @ centered / (n - 1)  # covariance sampel (unbiased)
        if self.covariance_type == "diag":
            cov = np.diag(np.diag(cov))
        # Regularisasi relatif terhadap rata-rata varians agar skala fitur tidak berpengaruh.
        cov = cov + self.reg * np.trace(cov) / dim * np.eye(dim)
        self.cov = cov
        # Cholesky: cov = L L^T. Dipakai untuk log|cov| dan jarak Mahalanobis.
        self._chol = np.linalg.cholesky(cov)
        self.logdet = 2.0 * np.sum(np.log(np.diag(self._chol)))
        self.n_frames = n
        return self

    def frame_nll(self, frames):
        # -log N(x; mean, cov) = 0.5 * (D log 2pi + log|cov| + (x-mean)^T cov^-1 (x-mean))
        frames = np.atleast_2d(np.asarray(frames, dtype=np.float64))
        diff = frames - self.mean
        z = np.linalg.solve(self._chol, diff.T)  # L z = (x - mean)
        mahalanobis = np.sum(z ** 2, axis=0)
        return 0.5 * (frames.shape[1] * LOG_2PI + self.logdet + mahalanobis)

    def nll(self, frames):
        # NLL satu ucapan dinormalisasi per frame (rata-rata), supaya cost antar ucapan
        # yang panjangnya berbeda bisa dibandingkan. Urutan model (argmin) sama dengan
        # memakai jumlah NLL semua frame.
        return float(np.mean(self.frame_nll(frames)))


class GaussianVowelRecognizer:
    # Kumpulan Gaussian, satu untuk tiap vowel.

    def __init__(self, covariance="full", reg=DEFAULT_REG):
        self.covariance = covariance
        self.reg = reg
        self.models = {}

    def fit(self, records):
        # records: hasil extract_features(), tiap record punya "label" dan "mfcc".
        frames = {}
        for record in records:
            frames.setdefault(record["label"], []).append(record["mfcc"])
        labels = [v for v in VOWEL_ORDER if v in frames] + sorted(set(frames) - set(VOWEL_ORDER))
        self.models = {label: Gaussian(self.covariance, self.reg).fit(np.vstack(frames[label]))
                       for label in labels}
        return self

    def scores(self, features):
        return {label: model.nll(features) for label, model in self.models.items()}

    def predict(self, features):
        scores = self.scores(features)
        return min(scores, key=scores.get), scores


def run_experiments(data_dir=DEFAULT_DATA_DIR, covariance="full", cmvn=False, reg=DEFAULT_REG):
    # Latih dari template, uji pada closed recordings, closed speakers, dan open speakers.
    data = load_dataset(data_dir)
    template = extract_features(data["template"], cmvn=cmvn)
    scenarios = {
        "closed_recordings": template,  # rekaman yang sama dengan data latih
        "closed_speakers": extract_features(data["closed"], cmvn=cmvn),
        "open_speakers": extract_features(load_open_set(data_dir), cmvn=cmvn),
    }
    recognizer = GaussianVowelRecognizer(covariance, reg).fit(template)
    results = {}
    for name, records in scenarios.items():
        outputs = [recognizer.predict(r["mfcc"]) for r in records]
        result = evaluate(records, [label for label, _ in outputs])
        for row, (_, scores) in zip(result["rows"], outputs):
            row["nll"] = scores
        results[name] = result
    accuracies = [results[name]["accuracy"] for name in scenarios]
    config = {"covariance": covariance, "cmvn": cmvn, "reg": reg,
              "train_frames": {k: m.n_frames for k, m in recognizer.models.items()}}
    return {"config": config, "results": results,
            "average_accuracy": float(np.mean(accuracies)), "recognizer": recognizer}


def save_outputs(run, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"config": run["config"], "average_accuracy": run["average_accuracy"],
               "results": {name: {k: v for k, v in res.items() if k != "rows"}
                           for name, res in run["results"].items()}}
    (output_dir / "gaussian_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    labels = list(run["recognizer"].models)
    with open(output_dir / "gaussian_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "filename", "speaker", "label", "prediction", "correct"]
                        + [f"nll_{v}" for v in labels])
        for name, res in run["results"].items():
            for row in res["rows"]:
                writer.writerow([name, row["filename"], row["speaker"], row["label"],
                                 row["prediction"], int(row["correct"])]
                                + [f"{row['nll'][v]:.4f}" for v in labels])


def print_summary(run):
    cfg = run["config"]
    print(f"Model Gaussian (covariance={cfg['covariance']}, cmvn={cfg['cmvn']}, reg={cfg['reg']})")
    print("Frame latih per vowel:", cfg["train_frames"])
    for name, res in run["results"].items():
        per_vowel = ", ".join(f"{v}={s['accuracy']:.0f}%" for v, s in res["per_vowel"].items())
        print(f"\n{name}: {res['correct']}/{res['total']} = {res['accuracy']:.2f}%  ({per_vowel})")
        print(format_confusion(res))
    print(f"\nRata-rata akurasi 3 skenario: {run['average_accuracy']:.2f}%")


def run_ablation(data_dir=DEFAULT_DATA_DIR, reg=DEFAULT_REG, output_dir=None):
    # Eksperimen tambahan: full vs diagonal covariance, dengan dan tanpa CMVN per speaker.
    rows = []
    for covariance in ["full", "diag"]:
        for cmvn in [False, True]:
            run = run_experiments(data_dir, covariance, cmvn, reg)
            rows.append({"covariance": covariance, "cmvn": cmvn,
                         **{name: res["accuracy"] for name, res in run["results"].items()},
                         "average": run["average_accuracy"]})
            print(f"covariance={covariance:<4} cmvn={str(cmvn):<5} "
                  + "  ".join(f"{name}={res['accuracy']:.2f}%" for name, res in run["results"].items())
                  + f"  rata-rata={run['average_accuracy']:.2f}%")
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "gaussian_ablation.json").write_text(json.dumps(rows, indent=2),
                                                                 encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Sistem 2: pengenalan vowel dengan Model Gaussian")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--covariance", choices=["full", "diag"], default="full")
    parser.add_argument("--cmvn", action="store_true", help="CMVN per speaker sebelum pemodelan")
    parser.add_argument("--reg", type=float, default=DEFAULT_REG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR.parents[1] / "outputs")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--ablation", action="store_true",
                        help="Bandingkan full/diag covariance dengan dan tanpa CMVN")
    args = parser.parse_args()
    if args.ablation:
        run_ablation(args.data_dir, args.reg, None if args.no_save else args.output_dir)
        return 0
    run = run_experiments(args.data_dir, args.covariance, args.cmvn, args.reg)
    print_summary(run)
    if not args.no_save:
        save_outputs(run, args.output_dir)
        print(f"Hasil disimpan ke {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
