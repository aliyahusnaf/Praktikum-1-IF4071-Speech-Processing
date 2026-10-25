import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data_loader import load_dataset, load_open_set
from .evaluation import VOWEL_ORDER, evaluate, format_confusion
from .mfcc import extract_features
from .preprocessing import DEFAULT_DATA_DIR

DEFAULT_ITERATIONS = 5
STEPS = [(1, 1, 2.0), (1, 0, 1.0), (0, 1, 1.0)]


def local_cost(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sq = np.sum(x ** 2, axis=1)[:, None] + np.sum(y ** 2, axis=1)[None, :] - 2 * x @ y.T
    return np.sqrt(np.maximum(sq, 0.0))


def dtw(x, y, return_path=False):
    cost = local_cost(x, y)
    n, m = cost.shape
    if n == 0 or m == 0:
        raise ValueError("Sekuens fitur tidak boleh kosong")
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    move = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = cost[i - 1, j - 1]
            candidates = [acc[i - di, j - dj] + w * c for di, dj, w in STEPS]
            k = int(np.argmin(candidates))
            acc[i, j] = candidates[k]
            move[i, j] = k
    distance = float(acc[n, m] / (n + m))
    if not return_path:
        return distance
    path, i, j = [], n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        di, dj, _ = STEPS[move[i, j]]
        i, j = i - di, j - dj
    return distance, path[::-1]


def medoid(sequences):
    totals = [sum(dtw(a, b) for b in sequences) for a in sequences]
    return int(np.argmin(totals))


def average_template(sequences, iterations=DEFAULT_ITERATIONS):
    sequences = [np.asarray(s, dtype=np.float64) for s in sequences]
    template = sequences[medoid(sequences)].copy()
    for _ in range(iterations):
        sums = np.zeros_like(template)
        counts = np.zeros(len(template))
        for seq in sequences:
            _, path = dtw(template, seq, return_path=True)
            for i, j in path:
                sums[i] += seq[j]
                counts[i] += 1
        template = sums / counts[:, None]
    return template


class DTWVowelRecognizer:
    def __init__(self, iterations=DEFAULT_ITERATIONS):
        self.iterations = iterations
        self.templates = {}

    def fit(self, records):
        sequences = {}
        for record in records:
            sequences.setdefault(record["label"], []).append(record["mfcc"])
        labels = [v for v in VOWEL_ORDER if v in sequences] + sorted(set(sequences) - set(VOWEL_ORDER))
        self.templates = {label: average_template(sequences[label], self.iterations)
                          for label in labels}
        return self

    def scores(self, features):
        return {label: dtw(features, template) for label, template in self.templates.items()}

    def predict(self, features):
        scores = self.scores(features)
        return min(scores, key=scores.get), scores


def run_experiments(data_dir=DEFAULT_DATA_DIR, cmvn=False, iterations=DEFAULT_ITERATIONS):
    data = load_dataset(data_dir)
    template = extract_features(data["template"], cmvn=cmvn)
    scenarios = {
        "closed_recordings": template,
        "closed_speakers": extract_features(data["closed"], cmvn=cmvn),
        "open_speakers": extract_features(load_open_set(data_dir), cmvn=cmvn),
    }
    recognizer = DTWVowelRecognizer(iterations).fit(template)
    results = {}
    for name, records in scenarios.items():
        outputs = [recognizer.predict(r["mfcc"]) for r in records]
        result = evaluate(records, [label for label, _ in outputs])
        for row, (_, scores) in zip(result["rows"], outputs):
            row["cost"] = scores
        results[name] = result
    accuracies = [results[name]["accuracy"] for name in scenarios]
    config = {"cmvn": cmvn, "iterations": iterations,
              "template_frames": {k: len(t) for k, t in recognizer.templates.items()}}
    return {"config": config, "results": results,
            "average_accuracy": float(np.mean(accuracies)), "recognizer": recognizer}


def save_outputs(run, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"config": run["config"], "average_accuracy": run["average_accuracy"],
               "results": {name: {k: v for k, v in res.items() if k != "rows"}
                           for name, res in run["results"].items()}}
    (output_dir / "dtw_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    labels = list(run["recognizer"].templates)
    with open(output_dir / "dtw_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "filename", "speaker", "label", "prediction", "correct"]
                        + [f"cost_{v}" for v in labels])
        for name, res in run["results"].items():
            for row in res["rows"]:
                writer.writerow([name, row["filename"], row["speaker"], row["label"],
                                 row["prediction"], int(row["correct"])]
                                + [f"{row['cost'][v]:.4f}" for v in labels])


def print_summary(run):
    cfg = run["config"]
    print(f"DTW averaged template (cmvn={cfg['cmvn']}, iterations={cfg['iterations']})")
    print("Frame template per vowel:", cfg["template_frames"])
    for name, res in run["results"].items():
        per_vowel = ", ".join(f"{v}={s['accuracy']:.0f}%" for v, s in res["per_vowel"].items())
        print(f"\n{name}: {res['correct']}/{res['total']} = {res['accuracy']:.2f}%  ({per_vowel})")
        print(format_confusion(res))
    print(f"\nRata-rata akurasi 3 skenario: {run['average_accuracy']:.2f}%")


def run_ablation(data_dir=DEFAULT_DATA_DIR, output_dir=None):
    rows = []
    for iterations in [0, 1, DEFAULT_ITERATIONS]:
        for cmvn in [False, True]:
            run = run_experiments(data_dir, cmvn, iterations)
            rows.append({"iterations": iterations, "cmvn": cmvn,
                         **{name: res["accuracy"] for name, res in run["results"].items()},
                         "average": run["average_accuracy"]})
            print(f"iterations={iterations} cmvn={str(cmvn):<5} "
                  + "  ".join(f"{name}={res['accuracy']:.2f}%" for name, res in run["results"].items())
                  + f"  rata-rata={run['average_accuracy']:.2f}%")
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "dtw_ablation.json").write_text(json.dumps(rows, indent=2),
                                                            encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Sistem 1: pengenalan vowel dengan DTW")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cmvn", action="store_true", help="CMVN per speaker sebelum DTW")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                        help="Jumlah iterasi averaging (0 = medoid saja)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR.parents[1] / "outputs")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--ablation", action="store_true",
                        help="Bandingkan jumlah iterasi averaging dengan dan tanpa CMVN")
    args = parser.parse_args()
    if args.ablation:
        run_ablation(args.data_dir, None if args.no_save else args.output_dir)
        return 0
    run = run_experiments(args.data_dir, args.cmvn, args.iterations)
    print_summary(run)
    if not args.no_save:
        save_outputs(run, args.output_dir)
        print(f"Hasil disimpan ke {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
